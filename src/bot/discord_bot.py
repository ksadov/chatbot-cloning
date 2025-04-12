import asyncio
import json
from pathlib import Path
from typing import Optional

import discord

from src.bot.chat_controller import ChatController
from src.bot.message import Message
from src.message_database.utils import database_from_config_path
from src.utils.local_logger import LocalLogger

intents = discord.Intents.default()
intents.message_content = True


class DiscordBot(discord.Client):
    def __init__(
        self,
        bot_config_path: Path,
        discord_config_path: Path,
        database_config_path: Optional[Path],
        logger: LocalLogger,
    ):
        super().__init__(intents=intents)
        self.chat_controller = ChatController(
            bot_config_path,
            logger,
        )
        self.discord_config = json.load(open(discord_config_path))
        self.logger = logger
        self.database = (
            database_from_config_path(database_config_path)
            if database_config_path
            else None
        )

    async def on_ready(self):
        self.logger.info(f"{self.user} has connected to Discord!")

    def message_from_discord_message(self, message: discord.Message) -> Message:
        new_message = Message(
            conversation=message.channel.id,
            platform="discord",
            sender_name=message.author.name,
            text_content=message.content,
            timestamp=message.created_at,
            bot_config=self.discord_config,
            platform_specific_message_id=message.id,
            platform_specific_user_id=message.author.id,
            global_user_id=message.author.id,
            attachments=message.attachments,
            replies_to_message_id=(
                message.reference.message_id if message.reference else None
            ),
            reactions=message.reactions,
            server_nickname=message.author.nick,
            account_username=message.author.name,
        )
        if self.database:
            self.database.store_message(new_message)
            self.logger.debug(f"Message added to database: {new_message.id}")
        return new_message

    def can_answer(self, message: discord.Message) -> bool:
        is_dm = isinstance(message.channel, discord.DMChannel)
        if message.author == self.user:
            return False
        elif is_dm or message.channel.name in self.discord_config["channels"]:
            return True
        else:
            return False

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            self_message = self.message_from_discord_message(message)
            self.chat_controller.update_conv_history(self_message)
            return
        elif self.can_answer(message):
            user_message = self.message_from_discord_message(message)
            self.chat_controller.update_conv_history(user_message)
            self.logger.info(f"Received message: {message.content}")
            if message.content == self.discord_config["clear_command"]:
                self.conv_history.clear()
                self.logger.info(
                    f"Conversation history cleared for channel {message.channel.id}"
                )
                await message.channel.send("[Conversation history cleared]")
            else:
                prompt, responses = self.chat_controller.make_response(user_message)
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f"Responses: {responses}")
                # only send next response once previous response is sent
                for response in responses:
                    await message.channel.send(response)
                    await asyncio.sleep(1)
                self.logger.debug(f"Sent response to channel {message.channel.id}")

    async def on_error(self, event_method, *args):
        self.logger.error(f"Error in event {event_method}: {args}")

    def run(self):
        self.logger.info("Running Discord bot...")
        super().run(self.discord_config["token"])
