import asyncio
import json
from pathlib import Path
from typing import Optional

import discord

from src.bot.chat_controller import ChatController
from src.bot.message import Message, ReactionMessage
from src.message_database.utils import database_from_config_path
from src.utils.local_logger import LocalLogger

intents = discord.Intents.default()
intents.message_content = True


def get_displayed_name(user: discord.Member) -> str:
    if hasattr(user, "display_name"):
        return user.display_name
    elif hasattr(user, "global_name"):
        return user.global_name
    else:
        return user.name


def get_chat_name(message: discord.Message) -> str:
    if isinstance(message.channel, discord.DMChannel):
        if message.channel.recipient:
            recipient_names = [get_displayed_name(message.channel.recipient)]
        else:
            recipient_names = [
                get_displayed_name(user) for user in message.channel.recipients
            ]
        if recipient_names:
            chat_name = f"Discord DM with {', '.join(recipient_names)}"
        else:
            chat_name = "Discord DM"
    else:
        channel_name = message.channel.name
        server_name = message.guild.name
        chat_name = f"Discord conversation in channel {server_name} - {channel_name}"
    return chat_name


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
        self.response_tasks = {}  # conversation_id -> asyncio.Task

    async def on_ready(self):
        self.logger.info(f"{self.user} has connected to Discord!")

    async def get_referenced_message(self, message: discord.Message) -> discord.Message:
        if message.reference:
            if message.reference.cached_message:
                return message.reference.cached_message
            else:
                return await message.channel.fetch_message(message.reference.message_id)
        else:
            return None

    async def message_from_discord_message(self, message: discord.Message) -> Message:
        referenced_message = await self.get_referenced_message(message)
        if referenced_message:
            text_content = f"[Replying to {referenced_message.author.name}: {referenced_message.content}]\n\n{message.content}"
        else:
            text_content = message.content
        try:
            new_message = Message(
                conversation=get_chat_name(message),
                platform="discord",
                sender_name=get_displayed_name(message.author),
                text_content=text_content,
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
                server_nickname=(
                    message.author.nick if hasattr(message.author, "nick") else None
                ),
                account_username=message.author.name,
            )
        except Exception as e:
            self.logger.error(f"Error making message from discord message: {e}")
            self.logger.error(f"Message: {message}")
            raise e
        if self.database:
            self.logger.debug(f"Storing message in database: {new_message.id}")
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
        conv_clear_message = "[Conversation history cleared]"
        try:
            if (
                message.author == self.user
                and not message.content == conv_clear_message
            ):
                self_message = await self.message_from_discord_message(message)
                self.chat_controller.update_conv_history(self_message)
                return
            elif self.can_answer(message):
                user_message = await self.message_from_discord_message(message)
                self.chat_controller.update_conv_history(user_message)
                self.logger.info(f"Received message: {message.content}")

                conversation_id = user_message.conversation

                # Cancel any previous response task for this conversation
                prev_task = self.response_tasks.get(conversation_id)
                if prev_task and not prev_task.done():
                    prev_task.cancel()
                    self.logger.info(
                        f"Cancelled previous response for {conversation_id}"
                    )

                # Start a new response task
                task = asyncio.create_task(
                    self.handle_response(message, user_message, conv_clear_message)
                )
                self.response_tasks[conversation_id] = task

        except Exception as e:
            self.logger.error(f"Error in on_message: {e}")
            raise e

    async def handle_response(self, message, user_message, conv_clear_message):
        try:
            if message.content == self.discord_config["clear_command"]:
                self.chat_controller.conv_history_dict[
                    user_message.conversation
                ].clear()
                self.logger.info(
                    f"Conversation history cleared for channel {message.channel.id}"
                )
                await message.channel.send(conv_clear_message)
            else:
                prompt, responses = self.chat_controller.make_response(user_message)
                self.logger.debug(f"Prompt: {prompt}")
                self.logger.debug(f"Responses: {responses}")
                for response in responses:
                    await message.channel.send(response)
                    await asyncio.sleep(0.5)
                self.logger.info(f"Sent response to channel {message.channel.id}")
        except asyncio.CancelledError:
            self.logger.info(f"Response task cancelled for {user_message.conversation}")
        except Exception as e:
            self.logger.error(f"Error in handle_response: {e}")
            raise e

    async def handle_reaction(self, payload):
        try:
            removed = payload.event_type == "REACTION_REMOVE"
            reaction_author = await self.fetch_user(payload.user_id)
            original_message_author = await self.fetch_user(payload.message_author_id)
            print("channel id", payload.channel_id)
            channel = self.get_channel(payload.channel_id)
            if channel:
                original_discord_message = await channel.fetch_message(
                    payload.message_id
                )
            else:
                # we are in a dm
                # this breaks for removing reacts in dms unfortunately
                # so we may want to move to using on_reaction_add and on_reaction_remove
                # instead of on_raw_reaction_add and on_raw_reaction_remove
                target = (
                    original_message_author
                    if not original_message_author == self.user
                    else reaction_author
                )
                dm_channel = await self.create_dm(target)
                original_discord_message = await dm_channel.fetch_message(
                    payload.message_id
                )
            original_message = await self.message_from_discord_message(
                original_discord_message
            )
            reaction_author = await self.fetch_user(payload.user_id)
            reaction_author_name = get_displayed_name(reaction_author)
            chat_name = get_chat_name(original_discord_message)
            reaction_message = ReactionMessage(
                conversation=chat_name,
                timestamp=original_discord_message.created_at,
                original_message=original_message,
                removed=removed,
                sender_name=reaction_author_name,
                platform="discord",
                reaction=payload.emoji.name,
                bot_config=self.discord_config,
                platform_specific_user_id=reaction_author.id,
            )
            self.chat_controller.update_conv_history(reaction_message)
        except Exception as e:
            self.logger.error(f"Error in handle_reaction: {e}")
            raise e

    async def on_raw_reaction_add(self, payload):
        await self.handle_reaction(payload)
        self.logger.info(f"Reaction added: {payload}")

    async def on_raw_reaction_remove(self, payload):
        await self.handle_reaction(payload)
        self.logger.info(f"Reaction removed: {payload}")

    async def on_error(self, event_method, *args):
        self.logger.error(f"Error in event {event_method}: {args}")

    def run(self):
        self.logger.info("Running Discord bot...")
        super().run(self.discord_config["token"])
