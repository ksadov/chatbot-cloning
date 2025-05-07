import datetime
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel


class Reaction(BaseModel):
    emote: str
    count: int
    users_ids: List[str]


class Message:
    def __init__(
        self,
        conversation: str,
        timestamp: Optional[datetime.datetime],
        sender_name: str,
        platform: str,
        text_content: str,
        bot_config: Dict[str, str],
        platform_specific_user_id: Optional[str] = None,
        global_user_id: Optional[str] = None,
        server_nickname: Optional[str] = None,
        account_username: Optional[str] = None,
        attachments: [Dict[str, str]] = [],
        id: Optional[str] = None,
        platform_specific_message_id: Optional[str] = None,
        replies_to_message_id: Optional[str] = None,
        reactions: Optional[List[Reaction]] = None,
    ):
        self.conversation = conversation
        self.timestamp = timestamp
        self.sender_name = sender_name
        self.platform = platform
        self.platform_specific_user_id = platform_specific_user_id
        self.text_content = text_content
        self.bot_config = bot_config
        self.global_user_id = global_user_id
        self.server_nickname = server_nickname
        self.account_username = account_username
        self.attachments = attachments
        self.platform_specific_message_id = platform_specific_message_id
        self.replies_to_message_id = replies_to_message_id
        self.reactions = reactions
        self.id = (
            platform_specific_message_id
            if platform_specific_message_id
            else str(uuid.uuid4())
        )

    def attachments_str(self):
        if self.attachments:
            return "Attachments: " + ", ".join(
                [
                    f"{attachment['filename']} ({attachment['url']})"
                    for attachment in self.attachments
                ]
            )
        else:
            return ""

    def __str__(self):
        return (
            f"Message(conversation={self.conversation}, user_id={self.sender_name}, "
            f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M') if self.timestamp else 'None'}, "
            f"content={self.text_content} attachments={self.attachments})"
        )

    def rag_string(self, include_timestamp: bool):
        if include_timestamp and self.timestamp:
            return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.sender_name}: {self.text_content}\n {self.attachments_str()}"
        else:
            return f"{self.sender_name}: {self.text_content}\n {self.attachments_str()}"


class ReactionMessage(Message):
    def __init__(
        self,
        conversation: str,
        timestamp: Optional[datetime.datetime],
        original_message: Message,
        removed: bool,
        sender_name: str,
        platform: str,
        reaction: str,
        bot_config: Dict[str, str],
        platform_specific_user_id: Optional[str] = None,
        global_user_id: Optional[str] = None,
        server_nickname: Optional[str] = None,
        account_username: Optional[str] = None,
    ):
        if removed:
            text_content = f"[Removed reaction {reaction} from {original_message.rag_string(False)}]"
        else:
            text_content = (
                f"[Reacted with {reaction} to {original_message.rag_string(False)}]"
            )
        super().__init__(
            conversation,
            timestamp,
            sender_name,
            platform,
            text_content,
            bot_config,
            platform_specific_user_id,
            global_user_id,
            server_nickname,
            account_username,
        )

    def __str__(self):
        return f"ReactionMessage(conversation={self.conversation}, user_id={self.sender_name}, timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M') if self.timestamp else 'None'}, content={self.text_content})"
