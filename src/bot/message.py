import datetime
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
        timestamp: datetime.datetime,
        sender_name: str,
        platform: str,
        text_content: str,
        bot_config: Dict[str, str],
        platform_specific_user_id: Optional[str] = None,
        global_user_id: Optional[str] = None,
        server_nickname: Optional[str] = None,
        account_username: Optional[str] = None,
        attachments: Optional[Dict[str, str]] = None,
        platform_specific_message_id: Optional[str] = None,
        replies_to_message_id: Optional[str] = None,
        reactions: Optional[List[Reaction]] = None,
    ):
        self.conversation = conversation
        self.timestamp = timestamp
        self.sender_name = sender_name
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

    def __str__(self):
        return f"Message(conversation={self.conversation}, user_id={self.sender_name}, timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')}, content={self.text_content})"

    def rag_string(self, include_timestamp: bool):
        if include_timestamp:
            return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.sender_name}: {self.text_content}"
        else:
            return f"{self.sender_name}: {self.text_content}"
