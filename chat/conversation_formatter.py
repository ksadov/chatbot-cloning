from pathlib import Path
from jinja2 import Template
from typing import List, Tuple
from datetime import datetime as dt


class ConversationPromptFormatter:

    def __init__(self, template_path: Path) -> None:
        self.template = Template(template_path.read_text())

    def make_query(
        self,
        name: str,
        chat_user_name: str,
        conv_history: str,
        gt_results: List[str],
        conversation_results: List[str],
        include_timestamp: bool,
    ) -> str:
        context = {
            "name": name,
            "gt_results": gt_results,
            "conversation_results": conversation_results,
            "conv_history": conv_history,
            "include_timestamp": include_timestamp,
            "timestamp": (
                dt.now().strftime("%Y-%m-%d %H:%M") if include_timestamp else ""
            ),
        }
        return self.template.render(**context)

    def cleanup_output(
        self, output: str, target_name: str, chat_user_name: str
    ) -> list[str]:
        # trim everything after the first instance of responder name, if there is one
        output_trimmed = output.split(chat_user_name)[0]
        # sometimes output is multiple messages, split by newlines with the prefix target_name
        output_trimmed = output_trimmed.split(f"{target_name}:")
        # trim whitespace in front and back
        output_trimmed = [message.strip() for message in output_trimmed]
        return output_trimmed
