import datetime

import pandas as pd
import pyarrow as pa


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

    def to_pandas(self):
        dframe_dict = {
            "conversation": self.conversation,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "content": self.content,
        }
        dataframe = pd.DataFrame([dframe_dict])
        return dataframe

    def to_arrow(self):
        try:
            return pa.RecordBatch.from_pandas(self.to_pandas())
        except:
            raise

    def __str__(self):
        return f"Message(conversation={self.conversation}, user_id={self.user_id}, timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M')}, content={self.content})"

    def rag_string(self, include_timestamp: bool):
        if include_timestamp:
            return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M')}] {self.user_id}: {self.content}"
        else:
            return f"{self.user_id}: {self.content}"
