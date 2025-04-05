import argparse
import datetime
import torch
import json
import requests
from datetime import datetime as dt

from chat.llm_inference import LLM
from chat.utils import HiddenPrints, parse_json
from chat.conversation import ConvHistory, Message

# silence annoying ragatouille logging
import logging

logging.getLogger("ragatouille").setLevel(logging.CRITICAL)


class RagModule:
    def __init__(self, vector_store_endpoint):
        self.vector_store_endpoint = vector_store_endpoint

    def search(self, query):
        response = requests.post(
            f"{self.vector_store_endpoint}/api/search",
            json={"query": query, "n_results": 5},
        )
        response.raise_for_status()
        response_texts = [result["text"] for result in response.json()["results"]]
        return response_texts

    def update(self, query):
        response = requests.post(
            f"{self.vector_store_endpoint}/api/update",
            json={"query": query},
        )
        response.raise_for_status()


class ChatController:
    def __init__(self, bot_config_path, llm_config_path, k):
        print("Setting up chatbot...")
        self.config = json.load(open(bot_config_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag_module = RagModule(self.config["vector_store_endpoint"])
        self.llm_config = json.load(open(llm_config_path))
        self.llm = LLM(self.llm_config, self.device)
        self.conv_history = ConvHistory(
            self.config["include_timestamp"],
            self.config["conversation_history_depth"],
            self.config["update_index_every"],
        )

    def make_response(self, query, speaker, conversation_name):
        query_timestamp = datetime.datetime.now()
        self.conv_history.add(
            Message(conversation_name, query_timestamp, speaker, query)
        )
        if self.config["update_rag_index"]:
            self.conv_history.update_rag_index(self.rag_module)
        try:
            full_query = self.conv_history.str_of_depth(
                self.config["query_context_depth"]
            )
            results = self.rag_module.search(full_query)
        except RetrievalError as e:
            results = []
            print(f"Error retrieving documents: {e}")
        prompt, response = self.llm.chat_step(
            self.config["name"],
            speaker,
            self.config["description"],
            self.conv_history,
            results,
            self.config["include_timestamp"],
        )
        response_timestamp = datetime.datetime.now()
        self.conv_history.add(
            Message(
                conversation_name, response_timestamp, self.config["name"], response
            )
        )
        if self.config["update_rag_index"]:
            self.conv_history.update_rag_index(self.rag_module)
        return prompt, response


def chat_loop(bot_config_path, llm_config_path, show_prompt, k):
    controller = ChatController(bot_config_path, llm_config_path, k)
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, response = controller.make_response(query, "user", "conversation_name")
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        print(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_prompt", "-s", action="store_true", help="Print the system prompt"
    )
    parser.add_argument(
        "--bot_config_path",
        "-b",
        type=str,
        help="Path to the config file",
        default="configs/bot/zef.json",
    )
    parser.add_argument(
        "--llm_config_path",
        "-l",
        type=str,
        help="Path to the model config file",
        default="configs/llm/Mixtral-8x7B-v01.json",
    )
    parser.add_argument(
        "--k",
        default=3,
        type=int,
    )
    args = parser.parse_args()
    chat_loop(args.bot_config_path, args.llm_config_path, args.show_prompt, args.k)


if __name__ == "__main__":
    main()
