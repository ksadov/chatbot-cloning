import json
from pathlib import Path

import discord

from src.bot.chat_controller import ChatController

intents = discord.Intents.default()
intents.message_content = True


class DiscordBot(discord.Client):
    def __init__(self, llm_config_path: Path, discord_config_path: Path):
        super().__init__(intents=intents)
        self.chat_controller = ChatController(
            llm_config_path, discord_config_path, llm_config_path
        )
        self.discord_config = json.load(open(discord_config_path))

    async def on_ready(self):
        print(f"{self.user} has connected to Discord!")

    async def on_message(self, message: discord.Message):
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
