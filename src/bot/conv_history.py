import datetime

import pandas as pd
import pyarrow as pa

from src.bot.rag_module import RagModule
from src.utils.local_logger import LocalLogger


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


class ConvHistory:
    def __init__(
        self,
        include_timestamp: bool,
        max_char_length: int,
        update_chunk_length: int,
        rag_module: RagModule,
        logger: LocalLogger,
        qa_mode: bool,
    ):
        self.logger = logger
        self.include_timestamp = include_timestamp
        self.history = []
        self.max_char_length = max_char_length
        self.update_chunk_length = update_chunk_length
        self.removed_buffer = []
        self.rag_module = rag_module
        self.qa_mode = qa_mode

    def add(self, message: Message):
        self.logger.debug(f"Adding message to history: {message}")
        self.history.append(message)
        self.trim_history()

    def trim_history(self):
        if self.qa_mode:
            # only keep most recent message
            self.history = self.history[-1:]
        else:
            while len(str(self)) > self.max_char_length:
                removed_msg = self.history.pop(0)
                self.removed_buffer.append(removed_msg)
                # When buffer reaches chunk size, trigger update
                if len(self.removed_buffer) >= self.update_chunk_length:
                    self.logger.debug(
                        f"Triggering RAG module update because buffer size {len(self.removed_buffer)} >= {self.update_chunk_length}"
                    )
                    self._process_removed_buffer()

    def _process_removed_buffer(self):
        if not self.removed_buffer:
            return
        # Process all messages in the buffer, even if more than chunk_length
        chunk_str = self._buffer_to_string()
        self.removed_buffer = []
        if self.rag_module is not None:
            self.logger.debug(f"Updating RAG module with chunk: {chunk_str}")
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

    def str_of_depth(self, depth: int) -> str:
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

    def __str__(self) -> str:
        return self.str_of_depth(len(self.history))
