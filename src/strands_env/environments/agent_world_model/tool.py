from __future__ import annotations

import subprocess
from datetime import timedelta
from typing import Any, Literal, override

from mcp import ClientSession
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands.types.tools import ToolResultContent

from strands_env.core.mcp_tool import MCPToolAdapter


class AgentWorldModelTool(MCPToolAdapter):
    """MCP tool backed by a `ClientSession` (single-server, direct connection).

    If `server_proc` is provided, polls the process before each call
    to fail fast when the server has exited.
    """

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        session: ClientSession,
        *,
        server_proc: subprocess.Popen | None = None,
        timeout: timedelta | None = None,
    ):
        super().__init__(mcp_tool, timeout=timeout)
        self._session = session
        self._server_proc = server_proc

    @override
    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[ToolResultContent], Literal["success", "error"]]:
        """Execute tool via MCP session, failing fast if server process has exited."""
        if self._server_proc is not None:
            returncode = self._server_proc.poll()
            if returncode is not None:
                raise RuntimeError(f"Server process exited with code {returncode}")
        result = await self._session.call_tool(name, args, self._timeout)
        content = [ToolResultContent(text=item.text) for item in result.content if isinstance(item, TextContent)]
        status: Literal["success", "error"] = "error" if result.isError else "success"
        return content, status
