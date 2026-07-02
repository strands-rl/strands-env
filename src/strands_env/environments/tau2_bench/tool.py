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

"""Adapt a tau2 `Tool` to a Strands `AgentTool`."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolGenerator, ToolResult, ToolResultContent, ToolSpec, ToolUse
from typing_extensions import override

from . import _tau2

if TYPE_CHECKING:
    from ._tau2 import Tau2Environment
    from ._tau2 import Tool as Tau2Tool


class Tau2BenchTool(AgentTool):
    """Strands `AgentTool` dispatching via tau2's `Environment.get_response` (which runs `sync_tools`)."""

    def __init__(self, tool: Tau2Tool, env: Tau2Environment, requestor: Literal["assistant", "user"]):
        """Initialize a `Tau2BenchTool` instance."""
        super().__init__()
        self._tool = tool
        self._env = env
        self._requestor = requestor
        self._props = tool.params.model_json_schema().get("properties", {})

    @property
    def tool_name(self) -> str:
        """Return the tau2 tool name."""
        return self._tool.name

    @property
    def tool_spec(self) -> ToolSpec:
        """Build a Strands `ToolSpec`; description mirrors tau2's `short_desc + long_desc` join."""
        description = self._tool.short_desc
        if self._tool.long_desc:
            description = description + "\n\n" + self._tool.long_desc
        return {
            "name": self._tool.name,
            "description": description,
            "inputSchema": {"json": self._tool.params.model_json_schema()},
        }

    @property
    def tool_type(self) -> str:
        """Strands tool type identifier."""
        return "python"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Invoke via `Environment.get_response` so `sync_tools` runs after each call."""
        call = _tau2.ToolCall(
            id=tool_use["toolUseId"],
            name=self._tool.name,
            # fix: typeless tool-call formats deliver numeric ids as ints missing string-keyed lookups in tau2.
            arguments={
                k: str(v) if self._props.get(k, {}).get("type") == "string" and type(v) is int else v
                for k, v in tool_use["input"].items()
            },
            requestor=self._requestor,
        )
        response = await asyncio.to_thread(self._env.get_response, call)
        status: Literal["success", "error"] = "error" if response.error else "success"
        yield ToolResultEvent(
            ToolResult(
                status=status,
                toolUseId=tool_use["toolUseId"],
                content=[ToolResultContent(text=response.content)],
            )
        )
