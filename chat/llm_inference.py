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


def make_instruct_query(
    name: str, description: str, conv_history: str, rag_results: List[str]
) -> Tuple[str, str]:
    context_string = make_context_string(rag_results)
    system = f"You are playing the role of {name}, {description}."
    user = (
        f"You are chatting online. "
        f"Write your next response in the conversation, based on the following excerpts"
        f"from {name}'s writing:{context_string}\n\n"
        f"Conversation history:\n\n{conv_history}\n{name}:"
    )
    return system, user


def make_completion_query(
    name: str,
    chat_user_name: str,
    description: str,
    conv_history: str,
    rag_results: List[str],
    include_timestamp: bool,
) -> str:
    context_string = make_context_string(rag_results)
    if include_timestamp:
        timestamp_str = f"[{dt.now().strftime('%Y-%m-%d %H:%M')}] "
    else:
        timestamp_str = ""
    prompt_str = (
        f"Character sheet:\n\n{name}: {description}.\n\n"
        f"Examples of {name}'s writing:{context_string}\n\n"
        f"Conversation history:\n\n{conv_history}\n{timestamp_str}{name}:"
    )
    return prompt_str


def cleanup_output(output: str, target_name: str, chat_user_name: str) -> str:
    # trim everything after the first instance of responder name, if there is one
    output_trimmed = output.split(chat_user_name)[0]
    # trim whitespace in front and back
    output_trimmed = output_trimmed.strip()
    # split by newline, take first line
    output_trimmed = output_trimmed.split("\n")[0]
    # trim before first :, if there is one
    colon = output_trimmed.find(f"{target_name}:") + len(target_name) + 1
    if colon != -1:
        output_trimmed = output_trimmed[colon + 1 :]
    # trim anything after the last period, unless there are no periods
    periods = [i for i, c in enumerate(output_trimmed) if c == "."]
    if len(periods) > 0:
        output_trimmed = output_trimmed[: periods[-1] + 1]
    # get rid of <s> and </s>
    output_trimmed = output_trimmed.replace("<s>", "")
    output_trimmed = output_trimmed.replace("</s>", "")
    # trim whitespace again
    output_trimmed = output_trimmed.strip()
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
        rag_results: List[str],
        include_timestamp: bool,
    ) -> Tuple[str, str]:
        if self.instruct:
            system, user = make_instruct_query(
                name, description, conv_history, rag_results
            )
            prompt = {"system": system, "user": user}
            response = self.make_instruct_request(system, user)
        else:
            prompt = make_completion_query(
                name,
                chat_user_name,
                description,
                conv_history,
                rag_results,
                include_timestamp,
            )
            response = self.make_completion_request(prompt, name, chat_user_name)
        return prompt, response

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
        return cleanup_output(raw_response, name, chat_user_name)
