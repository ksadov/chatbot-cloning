# modified from https://modelcontextprotocol.io/quickstart/client
from contextlib import AsyncExitStack
from typing import Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from pydantic import BaseModel

from src.bot.tools.types import Tool, ToolCallEvent
from src.utils.local_logger import LocalLogger


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: List[str]


class MCPClient:
    def __init__(self, logger: LocalLogger):
        self.logger = logger
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_config: MCPServerConfig):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
            server_name: Name of the server
            server_desc: Description of the server
        """
        self.server_name = server_config.name
        self.server_command = server_config.command
        self.server_args = server_config.args
        server_params = StdioServerParameters(
            command=self.server_command,
            args=self.server_args,
            env=None,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        self.logger.info(
            f"Connected to server {self.server_name} with tools: {[tool.name for tool in self.tools]}"
        )

    async def tool_call(self, tool_name: str, tool_args: dict) -> List[ToolCallEvent]:
        result = await self.session.call_tool(tool_name, tool_args)
        tool_call_events = []
        for result_content in result.content:
            if isinstance(result_content, TextContent):
                tool_call_events.append(
                    ToolCallEvent(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        tool_result=result_content.text,
                    )
                )
            else:
                raise ValueError(f"Unexpected tool call content type: {result_content}")
        return tool_call_events


async def get_mcp_tool_info(
    mcp_configs: List[MCPServerConfig],
    logger: LocalLogger,
) -> Tuple[List[Tool], Dict[str, MCPClient]]:
    tools = []
    tool_dict = {}
    for mcp_config in mcp_configs:
        mcp_client = MCPClient(logger)
        await mcp_client.connect_to_server(mcp_config)
        for tool in mcp_client.tools:
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                )
            )
            tool_dict[tool.name] = mcp_client
    return tools, tool_dict
