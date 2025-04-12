import datetime
import json
import sqlite3
from pathlib import Path
from typing import List

from src.bot.message import Message, Reaction
from src.message_database.interface import MessageDatabaseInterface


class SQLiteMessageDatabase(MessageDatabaseInterface):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    platform_specific_message_id TEXT UNIQUE,
                    conversation TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    sender_name TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    platform_specific_user_id TEXT,
                    global_user_id TEXT,
                    text_content TEXT NOT NULL,
                    bot_config TEXT,
                    server_nickname TEXT,
                    account_username TEXT,
                    attachments TEXT,
                    replies_to_message_id TEXT,
                    FOREIGN KEY (replies_to_message_id) REFERENCES messages(platform_specific_message_id)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reactions (
                    message_id TEXT,
                    emote TEXT,
                    count INTEGER,
                    users_ids TEXT,
                    PRIMARY KEY (message_id, emote),
                    FOREIGN KEY (message_id) REFERENCES messages(platform_specific_message_id)
                )
            """
            )

    def store_message(self, message: Message) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO messages (
                        id, platform_specific_message_id, conversation, timestamp, sender_name,
                        platform, platform_specific_user_id, global_user_id, text_content, bot_config,
                        server_nickname, account_username, attachments, replies_to_message_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message.id,
                        message.platform_specific_message_id,
                        message.conversation,
                        message.timestamp.isoformat(),
                        message.sender_name,
                        message.platform,
                        message.platform_specific_user_id,
                        message.global_user_id,
                        message.text_content,
                        json.dumps(message.bot_config),
                        message.server_nickname,
                        message.account_username,
                        (
                            json.dumps(message.attachments)
                            if message.attachments
                            else None
                        ),
                        message.replies_to_message_id,
                    ),
                )

                if message.reactions:
                    self.update_reactions(
                        message.platform_specific_message_id, message.reactions
                    )
                return True
            except sqlite3.IntegrityError as e:
                print(f"Integrity error: {e}")
                return False

    def update_message_content(self, message_id: str, new_content: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE messages
                SET text_content = ?
                WHERE platform_specific_message_id = ?
            """,
                (new_content, message_id),
            )
            return cursor.rowcount > 0

    def update_reactions(self, message_id: str, reactions: List[Reaction]) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reactions WHERE message_id = ?", (message_id,))

            for reaction in reactions:
                cursor.execute(
                    """
                    INSERT INTO reactions (message_id, emote, count, users_ids)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        message_id,
                        reaction.emote,
                        reaction.count,
                        json.dumps(reaction.users_ids),
                    ),
                )

            return True

    def recent_messages(self, limit: int = 10) -> List[Message]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM messages ORDER BY timestamp DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                # Convert row to dict and parse necessary fields
                row_dict = dict(row)
                row_dict["timestamp"] = datetime.datetime.fromisoformat(
                    row_dict["timestamp"]
                )
                row_dict["bot_config"] = (
                    json.loads(row_dict["bot_config"])
                    if row_dict["bot_config"]
                    else None
                )
                row_dict["attachments"] = (
                    json.loads(row_dict["attachments"])
                    if row_dict["attachments"]
                    else None
                )
                messages.append(Message(**row_dict))
            return messages


def test_sqlite_database():
    db_path = Path("zef_test_messages.sqlite")
    db = SQLiteMessageDatabase(db_path)
    # get recent messages
    messages = db.recent_messages(limit=10)
    for message in messages:
        print(str(message))


if __name__ == "__main__":
    test_sqlite_database()
