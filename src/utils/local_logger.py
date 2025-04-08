import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LocalLogger:

    def __init__(
        self,
        log_dir: Path,
        name,
        console_level,
        file_level,
    ):
        """Initialize the chatbot logger with both file and console output."""

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "chatbot.log"

        # Create handlers
        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )

        # Set levels
        console_handler.setLevel(logging.getLevelName(console_level))
        file_handler.setLevel(logging.getLevelName(file_level))

        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)
