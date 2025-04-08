from datetime import datetime as dt
from pathlib import Path
from typing import List

from jinja2 import Template


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

    def trim_chat_after_other_user(self, chat_log: str, target_name: str) -> str:
        """
        Trims a chat log after the first message not authored by the target user.

        Args:
            chat_log (str): The chat log text with format "{optional timestamp}{user}: {message}"
            target_name (str): The name of the target user whose messages we want to keep

        Returns:
            str: The trimmed chat log
        """
        lines = chat_log.split("\n")
        result = []

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Find where the username ends and the message begins
            colon_pos = line.find(":")
            if colon_pos == -1:
                result.append(line.strip())

            # Extract the username, handling optional timestamps
            username_part = line[:colon_pos].strip()
            # The username is the last word before the colon
            username = username_part.split()[-1] if username_part.split() else ""

            # Add the line to the result if it's from the target user
            if username == target_name:
                result.append(line)
            else:
                # Stop processing once we hit a message from another user
                break

        return "\n".join(result)

    def cleanup_output(self, output: str, target_name: str) -> list[str]:
        output_trimmed = self.trim_chat_after_other_user(output, target_name)
        # sometimes output is multiple messages, split by newlines with the prefix target_name
        output_trimmed = output_trimmed.split(f"{target_name}:")
        return output_trimmed
