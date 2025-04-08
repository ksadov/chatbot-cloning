import argparse
from pathlib import Path

from src.bot.discord_bot import DiscordBot
from src.utils.local_logger import LocalLogger


def main():
    # get discord token from environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bot_config_path",
        "-b",
        type=str,
        help="Path to the config file",
        default="configs/bot/zef_completion.json",
    )
    parser.add_argument(
        "--discord_config_path",
        "-d",
        type=str,
        help="Path to the discord config file",
        default="configs/discord/zef.json",
    )
    parser.add_argument(
        "--log_dir",
        "-l",
        type=Path,
        help="Log directory",
        default="logs",
    )
    parser.add_argument(
        "--file_log_level",
        "-f",
        type=str,
        help="File log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--console_log_level",
        "-c",
        type=str,
        help="Console log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args()
    logger = LocalLogger(
        args.log_dir, "discord_bot", args.console_log_level, args.file_log_level
    )
    discord_bot = DiscordBot(args.bot_config_path, args.discord_config_path, logger)
    discord_bot.run()


if __name__ == "__main__":
    main()
