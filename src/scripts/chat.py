import argparse
from pathlib import Path
from src.bot.chat_controller import ChatController


def chat_loop(bot_config_path: Path, show_prompt: bool = False):
    controller = ChatController(bot_config_path)
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
        default="configs/bot/zef_instruct.json",
    )
    args = parser.parse_args()
    chat_loop(args.bot_config_path, args.show_prompt)


if __name__ == "__main__":
    main()
