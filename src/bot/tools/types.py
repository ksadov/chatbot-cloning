from datetime import datetime
from typing import List, Optional

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


class Tool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

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


class ToolCallEvent(pydantic.BaseModel):
    tool_name: str
    tool_args: dict
    tool_result: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]

    def __str__(self):
        start_formatted = self.start_time.strftime("%H:%M:%S")
        if self.end_time:
            end_formatted = self.end_time.strftime("%H:%M:%S")
            time_formatted = f"{start_formatted} - {end_formatted}"
        else:
            time_formatted = start_formatted
        if self.tool_result:
            result_suffix = f"\n Result: {self.tool_result}"
        else:
            result_suffix = ""
        return f"* [{self.tool_name} with args {self.tool_args} at time {time_formatted}{result_suffix}]"


class ToolCallHistory(pydantic.BaseModel):
    tool_call_events: List[ToolCallEvent]
    max_length: int

    def add_event(self, event: ToolCallEvent):
        self.tool_call_events.append(event)
        if len(self.tool_call_events) > self.max_length:
            self.tool_call_events.pop(0)

    def __str__(self):
        if len(self.tool_call_events) == 0:
            return "No tool calls recorded yet."
        else:
            return "\n".join([str(event) for event in self.tool_call_events])


class ToolCallResponse(pydantic.BaseModel):
    tool_call_id: str
    tool_call_name: str
    tool_call_args: dict


class TextResponse(pydantic.BaseModel):
    text: str
