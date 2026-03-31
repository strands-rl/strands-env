"""MCP-Atlas tool adapter — calls the container's REST API."""

from __future__ import annotations

from typing import Literal

import httpx
from mcp.types import Tool as MCPToolDef
from strands.types.tools import ToolResultContent
from typing_extensions import override

from strands_env.tools.mcp_tool import MCPToolAdapter


class MCPAtlasTool(MCPToolAdapter):
    """MCP tool that calls the MCP-Atlas container's REST API."""

    def __init__(self, mcp_tool: MCPToolDef, http_client: httpx.AsyncClient, timeout: int = 60):
        """Initialize a `MCPAtlasTool` instance."""
        super().__init__(mcp_tool)
        self._http_client = http_client
        self._call_timeout = timeout

    @override
    async def call_tool(self, name: str, args: dict) -> tuple[list[ToolResultContent], Literal["success", "error"]]:
        """Execute tool via HTTP POST to the MCP-Atlas container."""
        response = await self._http_client.post(
            "/call-tool", json={"tool_name": name, "tool_args": args}, timeout=self._call_timeout
        )
        if response.status_code != 200:  # upstream MCP server error
            return [ToolResultContent(text=response.text)], "error"
        content = [
            ToolResultContent(text=item["text"] if isinstance(item, dict) and "text" in item else str(item))
            for item in response.json()
        ]
        return content, "success"
