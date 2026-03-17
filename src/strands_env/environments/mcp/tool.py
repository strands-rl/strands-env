# Copyright 2025-2026 Horizon RL Contributors
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

"""MCP tool base class for Strands agents.

`MCPTool` is an `AgentTool` subclass that handles tool spec building
and result parsing.  Subclasses override `_call_tool()` to provide the
actual call mechanism.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolGenerator, ToolResult, ToolSpec, ToolUse
from typing_extensions import override


class MCPTool(AgentTool):
    """Base MCP tool — shared spec and result parsing.

    Subclasses must implement `_call_tool()` to provide the call mechanism.
    """

    def __init__(
        self,
        mcp_tool: MCPToolDef,
        *,
        timeout: timedelta | None = None,
    ):
        """Initialize a `MCPTool` instance."""
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

    async def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute the tool call. Override in subclasses.

        Returns:
            An MCP `CallToolResult` (or compatible object with `.content`
            and `.isError` / `.is_error` attribute).
        """
        raise NotImplementedError

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Stream a tool result back to the agent. Delegates to `_call_tool()` 
        and parses the MCP result into a Strands `ToolResult`.
        """
        result = await self._call_tool(self._mcp_tool.name, tool_use["input"])
        content = [{"text": item.text} for item in result.content if isinstance(item, TextContent)]
        if not content:
            content = [{"text": str(item)} for item in result.content] or [{"text": ""}]
        is_error = getattr(result, "is_error", None) or getattr(result, "isError", False)
        status = "error" if is_error else "success"

        yield ToolResultEvent(ToolResult(status=status, toolUseId=tool_use["toolUseId"], content=content))
