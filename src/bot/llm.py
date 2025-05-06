import json
from pathlib import Path
from typing import List, Optional, Tuple

import pydantic
import requests

from src.bot.conversation_prompt_formatter import ConversationPromptFormatter
from src.bot.tools.communication import CommunicationTool
from src.bot.tools.tool_call_event import ToolCallHistory
from src.utils.local_logger import LocalLogger


class ToolCallResponse(pydantic.BaseModel):
    tool_call_id: str
    tool_call_name: str
    tool_call_args: dict


class TextResponse(pydantic.BaseModel):
    text: str


class LLM:
    def __init__(
        self,
        config: dict,
        prompt_template_path: Optional[Path],
        logger: LocalLogger,
    ):
        self.logger = logger
        self.config = config
        self.model_name = self.config["model"]
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
        current_conversation_name: str,
        tools: List[CommunicationTool],
        tool_call_history: ToolCallHistory,
    ) -> Tuple[str, List[TextResponse] | List[ToolCallResponse]]:
        prompt = self.conversation_formatter.make_query(
            name,
            chat_user_name,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
            current_conversation_name,
            tool_call_history,
        )
        if self.instruct:
            instruct_output = self.make_instruct_request(prompt, tools)
            if isinstance(instruct_output, TextResponse):
                responses = [instruct_output]
            else:
                responses = instruct_output
        else:
            responses = self.make_completion_request(prompt, name, chat_user_name)
        return prompt, responses

    def make_instruct_request(
        self, prompt: str, tools: list[str]
    ) -> TextResponse | List[ToolCallResponse]:
        """
        Make an instruct request to the LLM.
        If tools are provided, the response will be a list of ToolCallResponse.
        Otherwise, the response will be a TextResponse.
        """
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

        is_messages_endpoint = "messages" in self.api_base

        if tools:
            if is_messages_endpoint:
                tool_choice = {"type": "any"}
                formatted_tool_dicts = [
                    tool.message_api_representation() for tool in tools
                ]
            else:
                tool_choice = "required"
                formatted_tool_dicts = [
                    tool.completion_api_representation() for tool in tools
                ]

            request_body["tools"] = formatted_tool_dicts
            request_body["tool_choice"] = tool_choice

        response = requests.post(
            self.api_base,
            headers=headers,
            json=request_body,
        )
        response.raise_for_status()
        self.logger.debug(f"LLM response: {response.json()}")

        try:
            if is_messages_endpoint:
                content = response.json()["content"][0]
                if tools:
                    results = [
                        ToolCallResponse(
                            tool_call_id=content["id"],
                            tool_call_name=content["name"],
                            tool_call_args=content["input"],
                        )
                    ]
                else:
                    results = TextResponse(text=content["text"])
            else:
                if tools:
                    raw_results = response.json()["choices"][0]["message"]["tool_calls"]
                    print("raw_results", raw_results)
                    results = [
                        ToolCallResponse(
                            tool_call_id=result["id"],
                            tool_call_name=result["function"]["name"],
                            tool_call_args=json.loads(result["function"]["arguments"]),
                        )
                        for result in raw_results
                    ]
                else:
                    raw_results = response.json()["choices"][0]["message"]["content"]
                    results = TextResponse(text=raw_results)

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
        cleaned_response = self.conversation_formatter.cleanup_output(
            raw_response, name
        )
        return [TextResponse(text=response) for response in cleaned_response]
