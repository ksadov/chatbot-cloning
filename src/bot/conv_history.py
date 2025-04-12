from src.bot.message import Message
from src.bot.rag_module import RagModule
from src.utils.local_logger import LocalLogger


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

    def emergency_save(self):
        """Save the current history to the rag module"""
        if self.rag_module is not None:
            self.logger.debug("Saving current history to RAG module")
            # chunk current history into chunks of update_chunk_length
            chunks = [
                self.history[i : i + self.update_chunk_length]
                for i in range(0, len(self.history), self.update_chunk_length)
            ]
            # reverse the chunks so older chunks are processed first
            chunks.reverse()
            for chunk in chunks:
                chunk_str = "\n".join(
                    [
                        m.rag_string(include_timestamp=self.include_timestamp)
                        for m in chunk
                    ]
                )
                self.rag_module.update(chunk_str)

    def _buffer_to_string(self):
        return "\n".join(
            [
                self.rag_string(message, include_timestamp=self.include_timestamp)
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
