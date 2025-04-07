from bot.discord_bot import DiscordBot


def main():
    # get discord token from environment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bot_config_path",
        "-b",
        type=str,
        help="Path to the config file",
        default="configs/bot/zef.json",
    )
    parser.add_argument(
        "--discord_config_path",
        "-d",
        type=str,
        help="Path to the discord config file",
        default="configs/discord/zef.json",
    )
    parser.add_argument(
        "--llm_config_path",
        "-l",
        type=str,
        help="Path to the model config file",
        default="configs/llm/Mistal-7B-v01.json",
    )
    args = parser.parse_args()
    discord_config = parse_json(args.discord_config_path)
    chatbot = ChatBot(args.bot_config_path, discord_config, args.model_config)
    chatbot.run(discord_config["token"])


if __name__ == "__main__":
    main()
