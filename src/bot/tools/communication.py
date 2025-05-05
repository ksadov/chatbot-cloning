import json
from typing import List

import pydantic


class Property(pydantic.BaseModel):
    name: str
    type: str
    description: str


def input_schema_dict(properties: List[Property], required: List[str]) -> dict:
    properties_dict = {}
    for property in properties:
        properties_dict[property.name] = {
            "type": property.type,
            "description": property.description,
        }
    return {"type": "object", "properties": properties_dict, "required": required}


class CommunicationTool(pydantic.BaseModel):
    name: str
    description: str
    input_schema: dict
    required: list[str] = []

    def completion_api_representation(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def message_api_representation(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


REACT_TOOL = CommunicationTool(
    name="react",
    description="React to a message in the current conversation",
    input_schema=input_schema_dict(
        [
            Property(
                name="reaction",
                type="string",
                description="The reaction to send (must be a single emoji)",
            ),
            Property(
                name="identifying_substring",
                type="string",
                description="A subset of the message content that uniquely identifies the message. "
                "Example: 'Hello, world!' -> 'world'",
            ),
        ],
        ["reaction", "identifying_substring"],
    ),
)

MESSAGE_TOOL = CommunicationTool(
    name="message",
    description="Send a message in the current conversation",
    input_schema=input_schema_dict(
        [
            Property(
                name="message_content",
                type="string",
                description="The message content to send",
            ),
        ],
        ["message_content"],
    ),
)
DO_NOTHING_TOOL = CommunicationTool(
    name="do_nothing",
    description="Do nothing",
    input_schema=input_schema_dict([], []),
)
