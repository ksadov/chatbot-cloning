import datetime
import json
from pathlib import Path
from typing import Optional

from src.bot.conv_history import ConvHistory, Message
from src.bot.llm import LLM
from src.bot.rag_module import RagModule
from src.utils.local_logger import LocalLogger


class ChatController:

    def __init__(
        self,
        bot_config_path: Path,
        logger: LocalLogger,
        qa_mode: bool = False,
    ):
        self.logger = logger
        self.qa_mode = qa_mode
        with open(bot_config_path, "r") as f:
            self.config = json.load(f)
        self.target_name = self.config["name"]
        self.default_user_name = self.config["default_user_name"]
        self.device = "cpu"
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
        self.conv_history_dict = {}

    def update_conv_history(self, message: Message):
        if message.conversation not in self.conv_history_dict:
            self.conv_history_dict[message.conversation] = ConvHistory(
                self.config["include_timestamp"],
                self.config["max_conversation_length"],
                self.config["update_index_every"],
                (
                    self.conversation_rag_module
                    if self.config["update_rag_index"]
                    else None
                ),
                self.logger,
                self.qa_mode,
                message.conversation,
            )
        self.conv_history_dict[message.conversation].add(message)

    def make_response(
        self,
        message: Message,
    ) -> tuple[str, list[str]]:
        if message.conversation not in self.conv_history_dict:
            raise ValueError(f"Conversation {message.conversation} not found")
        full_query = self.conv_history_dict[message.conversation].str_of_depth(
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
        prompt, responses = self.llm.chat_step(
            self.target_name,
            message.sender_name,
            self.conv_history_dict[message.conversation],
            gt_results,
            conversation_results,
            self.config["include_timestamp"],
            message.conversation,
        )
        return prompt, responses

    def emergency_save(self):
        for conversation_name in self.conv_history_dict:
            if self.conversation_rag_module is not None:
                self.logger.info("Saving conversations to rag module")
                self.conv_history_dict[conversation_name].emergency_save()
