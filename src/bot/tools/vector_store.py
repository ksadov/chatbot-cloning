from datetime import datetime
from typing import List

from src.bot.rag_module import RagModule
from src.bot.tools.types import Property, Tool, ToolCallEvent, input_schema_dict


class VectorStoreTool(Tool):
    def __init__(self, rag_module: RagModule, is_gt: bool):
        if is_gt:
            name = "search_ground_truth"
            description = "Search the ground truth vector store database for content relevant to the query"
        else:
            name = "search_conversation"
            description = "Search the conversation vector store database for content relevant to the query"
        input_schema = input_schema_dict(
            [
                Property(
                    name="query",
                    type="string",
                    description="The query to search the vector store database for",
                ),
            ],
            ["query"],
        )
        super().__init__(name, description, input_schema)
        self.rag_module = rag_module

    def execute(self, query: str) -> List[ToolCallEvent]:
        start_time = datetime.now()
        results = self.rag_module.search(query)
        end_time = datetime.now()
        return [
            ToolCallEvent(
                tool_name=self.name,
                tool_args={"query": query},
                tool_result=result,
                start_time=start_time,
                end_time=end_time,
            )
            for result in results
        ]


def is_vector_store_tool(tool_name: str) -> bool:
    return tool_name in [
        "search_ground_truth",
        "search_conversation",
    ]
