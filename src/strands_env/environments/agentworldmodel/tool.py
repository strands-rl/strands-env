# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentWorldModel MCP tool — calls the per-task FastAPI server via MCP session."""

from __future__ import annotations

import subprocess
from datetime import timedelta
from typing import Any, Literal

from mcp import ClientSession
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands.types.tools import ToolResultContent
from typing_extensions import override

from strands_env.tools.mcp_tool import MCPToolAdapter


class AgentWorldModelMCPTool(MCPToolAdapter):
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
        """Initialize an `AgentWorldModelMCPTool` instance."""
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
