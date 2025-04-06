from typing import List, Tuple, Optional
from transformers import pipeline, AutoTokenizer
from datetime import datetime as dt
import requests
import json
import os
from pathlib import Path
from chat.conversation_formatter import ConversationPromptFormatter


class LLM:
    def __init__(self, config: dict, prompt_template_path: Optional[Path], device: str):
        self.config = config
        self.model_name = self.config["model"]
        self.device = device
        self.instruct = self.config["instruct"]
        self.api_base = self.config["api_base"]
        self.api_key = self.config["api_key"]
        self.prompt_params = self.config["prompt_params"]
        self.model = self.config["model"]
        if prompt_template_path:
            self.conversation_formatter = ConversationPromptFormatter(
                Path(prompt_template_path)
            )
        else:
            self.conversation_formatter = None

    def chat_step(
        self,
        name: str,
        chat_user_name: str,
        description: str,
        conv_history: str,
        gt_results: List[str],
        conversation_results: List[str],
        include_timestamp: bool,
    ) -> Tuple[str, List[str]]:
        prompt = self.conversation_formatter.make_query(
            name,
            chat_user_name,
            description,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
        )
        if self.instruct:
            response = self.make_instruct_request(user)
            # todo, handle multiple messages better when I get a good scaffold figured out
            responses = [response]
        else:
            raw_response = self.make_completion_request(prompt, name, chat_user_name)
            responses = self.conversation_formatter.cleanup_output(
                raw_response, name, chat_user_name
            )
        return prompt, responses

    def make_instruct_request(self, user: str) -> str:
        # use chat completion api: https://platform.openai.com/docs/api-reference/chat
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": user},
                ],
                **self.prompt_params,
            },
        )
        return response.json()["choices"][0]["message"]["content"]

    def make_completion_request(
        self, prompt: str, name: str, chat_user_name: str
    ) -> str:
        # use completion api: https://platform.openai.com/docs/api-reference/completions
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": prompt, **self.prompt_params},
        )
        raw_response = response.json()["choices"][0]["text"]
        return raw_response
