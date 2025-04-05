import pyarrow as pa
import pandas as pd


class Message:
    def __init__(self, conversation, timestamp, user_id, content):
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


class ConvHistory:
    def __init__(self, include_timestamp, max_length, update_every):
        self.include_timestamp = include_timestamp
        self.history = []
        self.max_length = max_length
        self.update_every = update_every
        self.update_counter = 0

    def add(self, message: Message):
        self.history.append(message)
        self.update_counter += 1
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def update_rag_index(self, rag_module):
        if self.update_counter >= self.update_every:
            self.update_counter = 0
            rag_module.update(str(self))

    def clear(self):
        self.history = []

    def str_of_depth(self, depth):
        if self.include_timestamp:
            return "\n".join(
                [
                    f"[{message.timestamp.strftime('%Y-%m-%d %H:%M')}] {message.user_id}: {message.content}"
                    for message in self.history[-depth:]
                ]
            )
        else:
            return "\n".join(
                [
                    f"{message.user_id}: {message.content}"
                    for message in self.history[-depth:]
                ]
            )

    def __str__(self):
        return self.str_of_depth(len(self.history))
