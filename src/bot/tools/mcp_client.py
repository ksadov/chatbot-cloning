# modified from https://modelcontextprotocol.io/quickstart/client
import json
from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(
        self, server_script_path: str, server_name: str, server_desc: str
    ):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
            server_name: Name of the server
            server_desc: Description of the server
        """
        self.server_name = server_name
        self.server_desc = server_desc
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
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
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def list_tool_strs(self) -> List[str]:
        response = await self.session.list_tools()
        tool_strs = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        return tool_strs

    async def get_server_info(self) -> str:
        tool_strs = await self.list_tool_strs()
        tool_strs_formatted = "\n".join(
            json.dumps(tool_str, indent=2) for tool_str in tool_strs
        )
        return f"{self.server_name}: {self.server_desc}\n\nAvailable tools:\n{tool_strs_formatted}"

    async def tool_call(self, tool_name: str, tool_args: dict) -> str:
        result = await self.session.call_tool(tool_name, tool_args)
        return result.content
