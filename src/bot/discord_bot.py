import json
from pathlib import Path

import discord

from src.bot.chat_controller import ChatController
from src.utils.local_logger import LocalLogger

intents = discord.Intents.default()
intents.message_content = True


class DiscordBot(discord.Client):
    def __init__(
        self, llm_config_path: Path, discord_config_path: Path, logger: LocalLogger
    ):
        super().__init__(intents=intents)
        self.chat_controller = ChatController(
            llm_config_path,
            logger,
        )
        self.discord_config = json.load(open(discord_config_path))
        self.logger = logger

    async def on_ready(self):
        self.logger.info(f"{self.user} has connected to Discord!")

    async def on_message(self, message: discord.Message):
        self.logger.info(f"Received message: {message.content}")
        if message.author == self.user:
            return
        elif message.channel.name in self.discord_config["channels"]:
            if message.content == self.discord_config["clear_command"]:
                self.conv_history.clear()
                self.logger.info(
                    f"Conversation history cleared for channel {message.channel.name}"
                )
                await message.channel.send("[Conversation history cleared]")
            else:
                prompt, response = self.chat_controller.make_response(
                    message.content, message.author.name, message.channel.name
                )
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f"Response: {response}")
                await message.channel.send(response)
                self.logger.debug(f"Sent response to channel {message.channel.name}")

    async def on_error(self, event_method, *args):
        self.logger.error(f"Error in event {event_method}: {args}")

    def run(self):
        self.logger.info("Running Discord bot...")
        super().run(self.discord_config["token"])
