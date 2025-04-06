from typing import List, Tuple
from transformers import pipeline, AutoTokenizer
from datetime import datetime as dt
import requests
import json


def make_context_string(rag_results: List[str]) -> str:
    context_string = ""
    for result in rag_results:
        context_string += f"\n\n- {result}"
    return context_string


def format_rag_results(gt_results: List[str], conversation_results: List[str]) -> str:
    gt_context_string = make_context_string(gt_results)
    gt_context_string = (
        f"Examples of target's writing:{gt_context_string}\n\n"
        if gt_context_string
        else ""
    )
    conversation_context_string = make_context_string(conversation_results)
    conversation_context_string = (
        f"Examples of previous simulated conversation:{conversation_context_string}\n\n"
        if conversation_context_string
        else ""
    )
    return gt_context_string, conversation_context_string


def make_instruct_query(
    name: str,
    description: str,
    conv_history: str,
    gt_results: List[str],
    conversation_results: List[str],
) -> Tuple[str, str]:
    gt_context_string, conversation_context_string = format_rag_results(
        gt_results, conversation_results
    )
    system = ""
    user = (
        f"You are simulating {name}, {description}, in an online conversation."
        f"Write the next response in the conversation, based on excerpts"
        f"{gt_context_string}"
        f"{conversation_context_string}"
        f"Conversation history:\n\n{conv_history}\n{name}:"
    )
    return system, user


def make_completion_query(
    name: str,
    chat_user_name: str,
    description: str,
    conv_history: str,
    gt_results: List[str],
    conversation_results: List[str],
    include_timestamp: bool,
) -> str:
    gt_context_string, conversation_context_string = format_rag_results(
        gt_results, conversation_results
    )
    if include_timestamp:
        timestamp_str = f"[{dt.now().strftime('%Y-%m-%d %H:%M')}] "
    else:
        timestamp_str = ""
    prompt_str = (
        f"Character sheet:\n\n{name}: {description}.\n\n"
        f"{gt_context_string}"
        f"{conversation_context_string}"
        f"Conversation history:\n\n{conv_history}\n{timestamp_str}{name}:"
    )
    return prompt_str


def cleanup_output(output: str, target_name: str, chat_user_name: str) -> list[str]:
    # trim everything after the first instance of responder name, if there is one
    output_trimmed = output.split(chat_user_name)[0]
    # sometimes output is multiple messages, split by newlines with the prefix target_name
    output_trimmed = output_trimmed.split(f"{target_name}:")
    # trim whitespace in front and back
    output_trimmed = [message.strip() for message in output_trimmed]
    return output_trimmed


class LLM:
    def __init__(self, config: dict, device: str):
        self.config = config
        self.model_name = self.config["model"]
        self.device = device
        self.instruct = self.config["instruct"]
        self.api_base = self.config["api_base"]
        self.api_key = self.config["api_key"]
        self.prompt_params = self.config["prompt_params"]
        self.model = self.config["model"]

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
        if self.instruct:
            system, user = make_instruct_query(
                name, description, conv_history, gt_results, conversation_results
            )
            prompt = {"system": system, "user": user}
            response = self.make_instruct_request(system, user)
            # todo, handle multiple messages better when I get a good scaffold figured out
            responses = [response]
        else:
            prompt = make_completion_query(
                name,
                chat_user_name,
                description,
                conv_history,
                gt_results,
                conversation_results,
                include_timestamp,
            )
            raw_response = self.make_completion_request(prompt, name, chat_user_name)
            responses = cleanup_output(raw_response, name, chat_user_name)
        return prompt, responses

    def make_instruct_request(self, system: str, user: str) -> str:
        # use chat completion api: https://platform.openai.com/docs/api-reference/chat
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
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
