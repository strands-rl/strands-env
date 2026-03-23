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

"""MCP tool adapter for Strands agents.

`MCPToolAdapter` is an `AgentTool` subclass that adapts an MCP tool definition
to the Strands agent interface.  It handles tool spec building; subclasses
implement `call_tool()` to provide the transport-specific call and result
parsing.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Literal

from mcp.types import Tool as MCPToolDef
from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolGenerator, ToolResult, ToolResultContent, ToolSpec, ToolUse
from typing_extensions import override


class MCPToolAdapter(AgentTool):
    """Adapts an MCP tool to the Strands `AgentTool` interface.

    Notes:
        Subclasses must implement `call_tool()` to execute the tool call and
        return parsed content and status.
    """

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        *,
        timeout: timedelta | None = None,
    ):
        """Initialize an `MCPToolAdapter` instance."""
        super().__init__()
        self._mcp_tool = mcp_tool
        self._timeout = timeout

    @property
    def tool_name(self) -> str:
        """Return the tool name."""
        return self._mcp_tool.name

    @property
    def tool_spec(self) -> ToolSpec:
        """Return the tool spec for the agent."""
        spec: ToolSpec = {
            "name": self._mcp_tool.name,
            "description": self._mcp_tool.description or self._mcp_tool.name,
            "inputSchema": {"json": self._mcp_tool.inputSchema},
        }
        if self._mcp_tool.outputSchema:
            spec["outputSchema"] = {"json": self._mcp_tool.outputSchema}
        return spec

    @property
    def tool_type(self) -> str:
        """Return the tool type identifier."""
        return "python"

    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[ToolResultContent], Literal["success", "error"]]:
        """Execute the tool call and return parsed results. Override in subclasses.

        Args:
            name: The tool name.
            args: The tool arguments.

        Returns:
            A tuple of (content, status).
        """
        raise NotImplementedError

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream a tool result back to the agent."""
        try:
            content, status = await self.call_tool(self._mcp_tool.name, tool_use["input"])
        except Exception as e:
            content = [ToolResultContent(text=f"Tool call failed: {type(e).__name__}: {e}")]
            status = "error"

        yield ToolResultEvent(ToolResult(status=status, toolUseId=tool_use["toolUseId"], content=content))
