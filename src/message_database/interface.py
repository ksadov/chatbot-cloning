from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from src.bot.message import Message, Reaction


class DatabaseType(Enum):
    SQLITE = "sqlite"
    # Future implementations could include:
    # POSTGRES = "postgres"
    # MONGODB = "mongodb"


class MessageDatabaseInterface(ABC):
    """Abstract base class defining the interface for message storage."""

    @abstractmethod
    def init_database(self) -> None:
        """Initialize the database with necessary tables/collections."""
        pass

    @abstractmethod
    def store_message(self, message: Message) -> bool:
        """Store a new message in the database."""
        pass

    @abstractmethod
    def update_message_content(self, message_id: str, new_content: str) -> bool:
        """Update the content of an existing message."""
        pass

    @abstractmethod
    def update_reactions(self, message_id: str, reactions: List[Reaction]) -> bool:
        """Update reactions for a message."""
        pass
