import datetime


class Message:
    def __init__(
        self,
        conversation: str,
        timestamp: datetime.datetime,
        user_id: str,
        content: str,
    ):
        self.conversation = conversation
        self.timestamp = timestamp
        self.user_id = user_id
        self.content = content

    def __str__(self):
        return f"Message(conversation={self.conversation}, user_id={self.user_id}, timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')}, content={self.content})"

    def rag_string(self, include_timestamp: bool):
        if include_timestamp:
            return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.user_id}: {self.content}"
        else:
            return f"{self.user_id}: {self.content}"
