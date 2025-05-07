from datetime import datetime
from typing import List, Optional

from src.bot.conv_history import ConvHistory
from src.bot.llm import LLM
from src.bot.rag_module import RagModule
from src.bot.tools.communication import (
    DO_NOTHING_TOOL,
    MESSAGE_TOOL,
    REACT_TOOL,
    REMOVE_REACT_TOOL,
    is_communication_tool,
)
from src.bot.tools.mcp_client import MCPServerConfig, get_mcp_tool_info
from src.bot.tools.types import ToolCallEvent, ToolCallHistory
from src.bot.tools.vector_store import VectorStoreTool, is_vector_store_tool
from src.utils.local_logger import LocalLogger


class Agent:
    def __init__(
        self,
        max_turns: int,
        mcp_server_configs: List[MCPServerConfig],
        llm: LLM,
        logger: LocalLogger,
        gt_rag_module: Optional[RagModule],
        conversation_rag_module: Optional[RagModule],
    ):
        self.max_turns = max_turns
        self.tools = [DO_NOTHING_TOOL, MESSAGE_TOOL, REACT_TOOL, REMOVE_REACT_TOOL]
        if gt_rag_module is not None:
            self.gt_vector_store_tool = VectorStoreTool(gt_rag_module, True)
            self.tools.append(self.gt_vector_store_tool)
        if conversation_rag_module is not None:
            self.conversation_vector_store_tool = VectorStoreTool(
                conversation_rag_module, False
            )
            self.tools.append(self.conversation_vector_store_tool)
        self.tool_mapping = {}
        self.tool_call_histories = {}
        self.turn_counter = 0
        self.mcp_server_configs = mcp_server_configs
        self.llm = llm
        self.logger = logger
        self.gt_rag_module = gt_rag_module
        self.conversation_rag_module = conversation_rag_module

    async def initialize_tools(self):
        mcp_tools, self.tool_mapping = await get_mcp_tool_info(
            self.mcp_server_configs,
            self.logger,
        )
        self.tools.extend(mcp_tools)

    async def invoke_agent(
        self,
        target_name: str,
        sender_name: str,
        conv_history: ConvHistory,
        gt_results: List[str],
        conversation_results: List[str],
        include_timestamp: bool,
        conversation: str,
    ):
        if self.turn_counter >= self.max_turns:
            allowed_tools = [MESSAGE_TOOL, REACT_TOOL]
        else:
            allowed_tools = self.tools
        if self.tool_call_histories.get(conversation, None) is None:
            self.tool_call_histories[conversation] = ToolCallHistory(
                tool_call_events=[],
                max_length=self.max_turns,
            )
        prompt, responses = self.llm.chat_step(
            target_name,
            sender_name,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
            conversation,
            allowed_tools,
            self.tool_call_histories[conversation],
        )
        for response in responses:
            if is_communication_tool(response.tool_call_name):
                self.tool_call_histories[conversation].add_event(
                    ToolCallEvent(
                        tool_name=response.tool_call_name,
                        tool_args=response.tool_call_args,
                        tool_result=None,
                        start_time=datetime.now(),
                        end_time=None,
                    )
                )
                self.turn_counter = 0
                return prompt, responses
            elif is_vector_store_tool(response.tool_call_name):
                if response.tool_call_name == "search_ground_truth":
                    tool_results = self.gt_vector_store_tool.execute(
                        response.tool_call_args["query"]
                    )
                else:
                    tool_results = self.conversation_vector_store_tool.execute(
                        response.tool_call_args["query"]
                    )
            else:
                mcp_server = self.tool_mapping.get(response.tool_call_name, None)
                if mcp_server is None:
                    raise ValueError(f"Tool {response.tool_call_name} not found")
                else:
                    tool_results = await mcp_server.tool_call(
                        response.tool_call_name, response.tool_call_args
                    )
            for tool_result in tool_results:
                self.tool_call_histories[conversation].add_event(tool_result)
        # since we didn't call a communication tool
        # we increment the turn counter and invoke the agent again
        self.turn_counter += 1
        return await self.invoke_agent(
            target_name,
            sender_name,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
            conversation,
        )
