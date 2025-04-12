import argparse
import datetime
import json
from pathlib import Path
from typing import Optional

from src.bot.chat_controller import ChatController
from src.bot.message import Message
from src.message_database.utils import database_from_config_path
from src.utils.local_logger import LocalLogger


def chat_loop(
    bot_config_path: Path,
    database_config_path: Optional[Path],
    show_prompt: bool,
    logger: LocalLogger,
):
    controller = ChatController(bot_config_path, logger)
    database = (
        database_from_config_path(database_config_path)
        if database_config_path
        else None
    )
    with open(bot_config_path, "r") as f:
        config = json.load(f)
    while True:
        query = input("> ")
        if query == "exit":
            controller.emergency_save()
            break
        message = Message(
            conversation="commandline_conversation",
            platform="commandline",
            sender_name="user",
            text_content=query,
            timestamp=datetime.datetime.now(),
            bot_config=config,
        )
        if database:
            database.store_message(message)
        controller.update_conv_history(message)
        prompt, responses = controller.make_response(message)
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        for response in responses:
            message = Message(
                conversation="commandline_conversation",
                platform="commandline",
                sender_name="bot",
                text_content=response,
                timestamp=datetime.datetime.now(),
                bot_config=config,
            )
            if database:
                database.store_message(message)
            print(response)
            controller.update_conv_history(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_prompt", "-s", action="store_true", help="Print the system prompt"
    )
    parser.add_argument(
        "--bot_config_path",
        "-b",
        type=Path,
        help="Path to the config file",
        default="configs/bot/zef_completion.json",
    )
    parser.add_argument(
        "--database_config_path",
        "-db",
        type=Path,
        help="Path to the database config file",
        default=None,
    )
    parser.add_argument(
        "--log_dir",
        "-l",
        type=Path,
        help="Path to the log directory",
        default="logs",
    )
    parser.add_argument(
        "--console_log_level",
        "-c",
        type=str,
        help="Console log level",
        default="INFO",
    )
    parser.add_argument(
        "--file_log_level",
        "-f",
        type=str,
        help="File log level",
        default="DEBUG",
    )
    args = parser.parse_args()
    logger = LocalLogger(
        args.log_dir, "chat", args.console_log_level, args.file_log_level
    )
    chat_loop(
        args.bot_config_path,
        args.database_config_path,
        args.show_prompt,
        logger,
    )


if __name__ == "__main__":
    main()
