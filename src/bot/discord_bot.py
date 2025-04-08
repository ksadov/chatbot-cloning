import asyncio
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

    def can_answer(self, message: discord.Message) -> bool:
        is_dm = isinstance(message.channel, discord.DMChannel)
        if message.author == self.user:
            return False
        elif is_dm or message.channel.name in self.discord_config["channels"]:
            print("can answer!")
            return True
        else:
            return False

    async def on_message(self, message: discord.Message):
        if self.can_answer(message):
            conversation_name = message.channel.id
            self.logger.info(f"Received message: {message.content}")
            if message.content == self.discord_config["clear_command"]:
                self.conv_history.clear()
                self.logger.info(
                    f"Conversation history cleared for channel {conversation_name}"
                )
                await message.channel.send("[Conversation history cleared]")
            else:
                prompt, responses = self.chat_controller.make_response(
                    message.content, message.author.name, conversation_name
                )
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f"Responses: {responses}")
                # only send next response once previous response is sent
                for response in responses:
                    await message.channel.send(response)
                    await asyncio.sleep(1)
                self.logger.debug(f"Sent response to channel {conversation_name}")

    async def on_error(self, event_method, *args):
        self.logger.error(f"Error in event {event_method}: {args}")

    def run(self):
        self.logger.info("Running Discord bot...")
        super().run(self.discord_config["token"])
