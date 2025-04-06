import argparse
import datetime
import torch
import json
import requests
from datetime import datetime as dt
from retrieval.embedding_core import RetrievalError
from chat.llm_inference import LLM
from chat.utils import HiddenPrints, parse_json
from chat.conversation import ConvHistory, Message

# silence annoying ragatouille logging
import logging


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
    def __init__(self, bot_config_path):
        print("Setting up chatbot...")
        self.config = json.load(open(bot_config_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gt_rag_module = (
            RagModule(self.config["gt_store_endpoint"])
            if self.config["gt_store_endpoint"]
            else None
        )
        self.conversation_rag_module = (
            RagModule(self.config["conversation_store_endpoint"])
            if self.config["conversation_store_endpoint"]
            else None
        )
        with open(self.config["llm_config"], "r") as f:
            self.llm_config = json.load(f)
        self.llm = LLM(
            self.llm_config, self.config["prompt_template_path"], self.device
        )
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
            self.conv_history.update_rag_index(self.conversation_rag_module)
        try:
            full_query = self.conv_history.str_of_depth(
                self.config["query_context_depth"]
            )
            if self.config["gt_store_endpoint"]:
                gt_results = self.gt_rag_module.search(full_query)
            else:
                gt_results = []
            if self.config["conversation_store_endpoint"]:
                conversation_results = self.conversation_rag_module.search(full_query)
            else:
                conversation_results = []
        except RetrievalError as e:
            results = []
            print(f"Error retrieving documents: {e}")
        prompt, responses = self.llm.chat_step(
            self.config["name"],
            speaker,
            self.conv_history,
            gt_results,
            conversation_results,
            self.config["include_timestamp"],
        )
        # stagger response timestamps by 1 second
        response_timestamps = [
            query_timestamp + datetime.timedelta(seconds=i)
            for i in range(len(responses))
        ]
        for response, response_timestamp in zip(responses, response_timestamps):
            self.conv_history.add(
                Message(
                    conversation_name, response_timestamp, self.config["name"], response
                )
            )
        if self.config["update_rag_index"]:
            self.conv_history.update_rag_index(self.conversation_rag_module)
        return prompt, responses


def chat_loop(bot_config_path, show_prompt):
    controller = ChatController(bot_config_path)
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, responses = controller.make_response(query, "user", "conversation_name")
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        for response in responses:
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
        default="configs/bot/zef_instruct.json",
    )
    args = parser.parse_args()
    chat_loop(args.bot_config_path, args.show_prompt)


if __name__ == "__main__":
    main()
