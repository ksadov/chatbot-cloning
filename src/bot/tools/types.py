from datetime import datetime
from typing import List

import pydantic


class Tool(pydantic.BaseModel):
    name: str
    description: str
    input_schema: dict

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
    tool_result: str
    start_time: datetime
    end_time: datetime

    def __str__(self):
        start_formatted = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_formatted = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        return f"{self.tool_name} ({start_formatted} - {end_formatted}): {self.tool_result}"


class ToolCallHistory(pydantic.BaseModel):
    tool_call_events: List[ToolCallEvent]
    max_length: int

    def add_event(self, event: ToolCallEvent):
        self.tool_call_events.append(event)
        if len(self.tool_call_events) > self.max_length:
            self.tool_call_events.pop(0)

    def __str__(self):
        return "\n".join([str(event) for event in self.tool_call_events])


class ToolCallResponse(pydantic.BaseModel):
    tool_call_id: str
    tool_call_name: str
    tool_call_args: dict


class TextResponse(pydantic.BaseModel):
    text: str
