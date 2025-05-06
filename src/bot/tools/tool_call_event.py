from typing import List

import pydantic


class ToolCallEvent(pydantic.BaseModel):
    tool_name: str
    tool_args: dict
    tool_result: str


class ToolCallHistory(pydantic.BaseModel):
    tool_call_events: List[ToolCallEvent]
    max_length: int

    def add_event(self, event: ToolCallEvent):
        self.tool_call_events.append(event)
        if len(self.tool_call_events) > self.max_length:
            self.tool_call_events.pop(0)

    def __str__(self):
        return "\n".join([str(event) for event in self.tool_call_events])
