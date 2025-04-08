import argparse
from pathlib import Path

from src.bot.chat_controller import ChatController
from src.utils.local_logger import LocalLogger


def chat_loop(
    bot_config_path: Path,
    show_prompt: bool,
    logger: LocalLogger,
):
    controller = ChatController(bot_config_path, logger)
    while True:
        query = input("> ")
        if query == "exit":
            break
        prompt, responses = controller.make_response(
            query, controller.default_user_name, "conversation"
        )
        if show_prompt:
            print("------------------")
            print("PROMPT:")
            print(prompt)
            print("------------------")
        for response in responses:
            print(response)


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
    chat_loop(args.bot_config_path, args.show_prompt, logger)


if __name__ == "__main__":
    main()
