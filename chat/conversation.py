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
    def __init__(
        self, include_timestamp, max_char_length, update_chunk_length, rag_module
    ):
        self.include_timestamp = include_timestamp
        self.history = []
        self.max_char_length = max_char_length
        self.update_chunk_length = update_chunk_length
        self.removed_buffer = []
        self.rag_module = rag_module

    def add(self, message: Message):
        self.history.append(message)
        self.trim_history()

    def trim_history(self):
        while len(str(self)) > self.max_char_length:
            removed_msg = self.history.pop(0)
            self.removed_buffer.append(removed_msg)
            # When buffer reaches chunk size, trigger update
            if len(self.removed_buffer) >= self.update_chunk_length:
                self._process_removed_buffer()

    def _process_removed_buffer(self):
        if not self.removed_buffer:
            return
        # Process all messages in the buffer, even if more than chunk_length
        chunk_str = self._buffer_to_string()
        self.removed_buffer = []
        if self.rag_module is not None:
            self.rag_module.update(chunk_str)
        return chunk_str

    def _buffer_to_string(self):
        if self.include_timestamp:
            return "\n".join(
                [
                    f"[{message.timestamp.strftime('%Y-%m-%d %H:%M')}] {message.user_id}: {message.content}"
                    for message in self.removed_buffer
                ]
            )
        else:
            return "\n".join(
                [
                    f"{message.user_id}: {message.content}"
                    for message in self.removed_buffer
                ]
            )

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
