import pyarrow as pa
import pandas as pd


class Message():
    def __init__(self, conversation, timestamp, author, content):
        self.conversation = conversation
        self.timestamp = timestamp
        self.author = author
        self.content = content

    def to_pandas(self):
        dframe_dict = {
            'conversation': self.conversation,
            'author': self.author,
            'timestamp': self.timestamp,
            'content': self.content,
        }
        dataframe = pd.DataFrame([dframe_dict])
        return dataframe

    def to_arrow(self):
        try:
            return pa.RecordBatch.from_pandas(self.to_pandas())
        except:
            raise

    def __str__(self):
        return (
            f"Message(conversation={self.conversation}, author={self.author}, timestamp={self.timestamp}, content={self.content})"
        )


class ConvHistory:
    def __init__(self, max_length=5):
        self.history = []
        self.max_length = max_length

    def add(self, message: Message):
        self.history.append(message)
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def clear(self):
        self.history = []

    def __str__(self):
        return "\n".join([f"{message.author}: {message.content}" for message in self.history])
