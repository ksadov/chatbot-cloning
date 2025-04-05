import argparse
import datetime
import torch
import json
from datetime import datetime as dt

from chat.llm_inference import setup_llm
from chat.utils import HiddenPrints, parse_json
from chat.conversation import ConvHistory, Message
from chat.retrieval import RAGModule, RetrievalError

# silence annoying ragatouille logging
import logging
logging.getLogger("ragatouille").setLevel(logging.CRITICAL)


class ChatController:
    def __init__(self, bot_config_path, llm_config_path, k):
        print("Setting up chatbot...")
        self.config = json.load(open(bot_config_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rag_module = RAGModule(self.config, k)
        self.llm = setup_llm(llm_config_path, self.device)
        with HiddenPrints():
            self.rag_module.search(query="warmup")
        self.conv_history = ConvHistory(
            self.config["include_timestamp"], self.config["conversation_history_depth"], self.config["update_index_every"]
        )

    def make_response(self, query, speaker, conversation_name):
        query_timestamp = datetime.datetime.now()
        self.conv_history.add(
            Message(conversation_name, query_timestamp, speaker, query))
        if self.config['update_rag_index']:
            self.conv_history.update_rag_index(self.rag_module)
        try:
            full_query = self.conv_history.str_of_depth(self.config['query_context_depth'])
            results = self.rag_module.search(query=full_query)
        except RetrievalError as e:
            results = []
            print(f"Error retrieving documents: {e}")
        prompt, response = self.llm.chat_step(
            self.config['name'], speaker, self.config['description'], self.conv_history, results, self.config['include_timestamp']
        )
        response_timestamp = datetime.datetime.now()
        self.conv_history.add(
            Message(conversation_name, response_timestamp, self.config['name'], response))
        if self.config['update_rag_index']:
            self.conv_history.update_rag_index(self.rag_module)
        return prompt, response


def chat_loop(bot_config_path, llm_config_path, show_prompt, k):
    controller = ChatController(bot_config_path, llm_config_path, k)
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, response = controller.make_response(
            query, "user", "conversation_name"
        )
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        print(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_prompt", "-s",
                        action="store_true", help="Print the system prompt")
    parser.add_argument('--bot_config_path', '-b', type=str,
                        help='Path to the config file', default='configs/bot/zef.json')
    parser.add_argument('--llm_config_path', '-l', type=str,
                        help='Path to the model config file', default='configs/llm/Mixtral-8x7B-v01.json')
    parser.add_argument("--k", default=3, type=int,)
    args = parser.parse_args()
    chat_loop(args.bot_config_path, args.llm_config_path, args.show_prompt, args.k)


if __name__ == "__main__":
    main()
