from typing import List

from src.bot.conv_history import ConvHistory
from src.bot.llm import LLM
from src.bot.tools.communication import DO_NOTHING_TOOL, MESSAGE_TOOL, REACT_TOOL
from src.bot.tools.mcp_client import MCPServerConfig, get_mcp_tool_info
from src.bot.tools.tool_call_event import ToolCallHistory
from src.utils.local_logger import LocalLogger


class Agent:
    def __init__(
        self,
        max_turns: int,
        mcp_server_configs: List[MCPServerConfig],
        llm: LLM,
        logger: LocalLogger,
    ):
        self.max_turns = max_turns
        self.tools = [DO_NOTHING_TOOL, MESSAGE_TOOL, REACT_TOOL]
        self.tool_mapping = {}
        self.tool_call_history = ToolCallHistory(
            tool_call_events=[],
            max_length=max_turns,
        )
        self.turn_counter = 0
        self.mcp_server_configs = mcp_server_configs
        self.llm = llm
        self.logger = logger

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
        prompt, responses = self.llm.chat_step(
            target_name,
            sender_name,
            conv_history,
            gt_results,
            conversation_results,
            include_timestamp,
            conversation,
            allowed_tools,
            self.tool_call_history,
        )
        for response in responses:
            mcp_server = self.tool_mapping.get(response.tool_call_name, None)
            if mcp_server is not None:
                tool_result = await mcp_server.tool_call(
                    response.tool_call_name, response.tool_call_args
                )
                self.tool_call_history.add_event(tool_result)
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
            else:
                self.turn_counter = 0
                return prompt, responses
