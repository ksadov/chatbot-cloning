from pathlib import Path
from typing import List, Optional, Tuple

import requests

from src.bot.conversation_prompt_formatter import ConversationPromptFormatter
from src.utils.local_logger import LocalLogger


class LLM:
    def __init__(
        self,
        config: dict,
        prompt_template_path: Optional[Path],
        device: str,
        logger: LocalLogger,
    ):
        self.logger = logger
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
        conv_history: str,
        gt_results: List[str],
        conversation_results: List[str],
        include_timestamp: bool,
    ) -> Tuple[str, List[str]]:
        prompt = self.conversation_formatter.make_query(
            name,
            chat_user_name,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
        )
        if self.instruct:
            response = self.make_instruct_request(prompt)
            # todo, handle multiple messages better when I get a good scaffold figured out
            responses = [response]
        else:
            raw_response = self.make_completion_request(prompt, name, chat_user_name)
            responses = self.conversation_formatter.cleanup_output(
                raw_response, name, chat_user_name
            )
        return prompt, responses

    def make_instruct_request(self, prompt: str) -> str:
        self.logger.debug(f"Making instruct request with prompt: {prompt}")
        is_anthropic = "claude" in self.model.lower() or "anthropic" in self.api_base

        headers = {"content-type": "application/json"}

        if is_anthropic:
            headers.update(
                {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
            )
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            **self.prompt_params,
        }

        response = requests.post(
            self.api_base,
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()
        self.logger.debug(f"LLM response: {response.json()}")

        is_messages_endpoint = "messages" in self.api_base

        try:
            if is_messages_endpoint:
                results = response.json()["content"][0]["text"]
            else:
                results = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Failed to parse API response: {str(e)}")

        return results

    def make_completion_request(
        self, prompt: str, name: str, chat_user_name: str
    ) -> str:
        self.logger.debug(f"Making completion request with prompt: {prompt}")
        # use completion api: https://platform.openai.com/docs/api-reference/completions
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "prompt": prompt, **self.prompt_params},
        )
        response.raise_for_status()
        self.logger.debug(f"LLM response: {response.json()}")
        raw_response = response.json()["choices"][0]["text"]
        return raw_response
