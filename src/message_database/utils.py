import json
from pathlib import Path

from src.message_database.interface import DatabaseType, MessageDatabaseInterface
from src.message_database.sqlite import SQLiteMessageDatabase


def create_database(db_type: DatabaseType, **kwargs) -> MessageDatabaseInterface:
    """Factory function to create the appropriate database instance."""
    if db_type == DatabaseType.SQLITE:
        db_path = kwargs.get("db_path")
        return SQLiteMessageDatabase(db_path)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def database_from_config_path(config_path: Path) -> MessageDatabaseInterface:
    with open(config_path, "r") as f:
        config = json.load(f)
    db_type = DatabaseType(config["database_type"])
    return create_database(db_type, **config["database_config"])
