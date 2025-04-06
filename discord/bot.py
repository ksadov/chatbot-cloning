import argparse
import discord
import os
import json

from chat.chat import ChatController

intents = discord.Intents.default()
intents.message_content = True


class ChatBot(discord.Client):
    def __init__(self, llm_config_path, discord_config_path, llm_config_path):
        super().__init__(intents=intents)
        self.chat_controller = ChatController(
            llm_config_path, discord_config_path, llm_config_path
        )
        self.discord_config = json.load(open(discord_config_path))

    async def on_ready(self):
        print(f"{self.user} has connected to Discord!")

    async def on_message(self, message):
        if message.author == self.user:
            return
        elif message.channel.name in self.discord_config["channels"]:
            if message.content == self.discord_config["clear_command"]:
                self.conv_history.clear()
                await message.channel.send("[Conversation history cleared]")
            else:
                prompt, response = self.chat_controller.make_response(
                    message.content, message.author.name, message.channel.name
                )
                print("------------------")
                print(prompt)
                print("------------------")
                await message.channel.send(response)


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
