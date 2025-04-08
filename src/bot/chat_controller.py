import datetime
import json
from pathlib import Path

import torch

from src.bot.conv_history import ConvHistory, Message
from src.bot.llm import LLM
from src.bot.rag_module import RagModule
from src.utils.local_logger import LocalLogger


class ChatController:

    def __init__(
        self, bot_config_path: Path, logger: LocalLogger, qa_mode: bool = False
    ):
        self.logger = logger
        self.qa_mode = qa_mode
        with open(bot_config_path, "r") as f:
            self.config = json.load(f)
        self.target_name = self.config["name"]
        self.default_user_name = self.config["default_user_name"]
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
            self.llm_config,
            self.config["prompt_template_path"],
            self.device,
            self.logger,
        )
        self.conv_history = ConvHistory(
            self.config["include_timestamp"],
            self.config["max_conversation_length"],
            self.config["update_index_every"],
            self.conversation_rag_module if self.config["update_rag_index"] else None,
            self.logger,
            self.qa_mode,
        )

    def make_response(
        self,
        query: str,
        speaker: str,
        conversation_name: str,
    ) -> tuple[str, list[str]]:
        self.logger.debug(f"Making response for query: {query}")
        query_timestamp = datetime.datetime.now()
        self.conv_history.add(
            Message(conversation_name, query_timestamp, speaker, query)
        )
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
        except Exception as e:
            gt_results = []
            conversation_results = []
            self.logger.error(f"Error retrieving documents: {e}")
        prompt, responses = self.llm.chat_step(
            self.target_name,
            speaker,
            self.conv_history,
            gt_results,
            conversation_results,
            self.config["include_timestamp"],
        )
        self.logger.debug(f"Generated prompt: {prompt}")
        self.logger.debug(f"Generated response: {responses}")
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
        return prompt, responses
