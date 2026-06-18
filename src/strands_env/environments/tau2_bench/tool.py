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
from typing import Any, Literal

from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolGenerator, ToolResult, ToolResultContent, ToolSpec, ToolUse
from typing_extensions import override


class Tau2BenchTool(AgentTool):
    """Strands `AgentTool` dispatching via tau2's `Environment.get_response` (which runs `sync_tools`)."""

    def __init__(self, tau2_tool: Any, tau2_env: Any, requestor: Literal["assistant", "user"]):
        """Initialize a `Tau2BenchTool` instance."""
        super().__init__()
        self._tau2_tool = tau2_tool
        self._tau2_env = tau2_env
        self._requestor: Literal["assistant", "user"] = requestor

    @property
    def tool_name(self) -> str:
        """Return the tau2 tool name."""
        return self._tau2_tool.name

    @property
    def tool_spec(self) -> ToolSpec:
        """Build a Strands `ToolSpec`; description mirrors tau2's `short_desc + long_desc` join."""
        description = self._tau2_tool.short_desc
        if self._tau2_tool.long_desc:
            description = description + "\n\n" + self._tau2_tool.long_desc
        return {
            "name": self._tau2_tool.name,
            "description": description,
            "inputSchema": {"json": self._tau2_tool.params.model_json_schema()},
        }

    @property
    def tool_type(self) -> str:
        """Strands tool type identifier."""
        return "python"

    def _coerce_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Stringify numbers passed to string-typed params.

        The GLM tool-call format is typeless, so a numeric id (`product_id`, `zip`) arrives as an
        `int` and tau2's string-keyed lookups miss. A function-calling API delivers args already
        typed per schema; mirror that by coercing against the tau2 param schema before dispatch.
        """
        props = self._tau2_tool.params.model_json_schema().get("properties", {})
        return {
            key: str(value)
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and props.get(key, {}).get("type") == "string"
            else value
            for key, value in args.items()
        }

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Invoke via `Environment.get_response` so `sync_tools` runs after each call."""
        from tau2.data_model.message import ToolCall  # type: ignore[import-not-found]

        call = ToolCall(
            id=tool_use["toolUseId"],
            name=self._tau2_tool.name,
            arguments=self._coerce_args(tool_use["input"]),
            requestor=self._requestor,
        )
        response = await asyncio.to_thread(self._tau2_env.get_response, call)
        status: Literal["success", "error"] = "error" if response.error else "success"
        yield ToolResultEvent(
            ToolResult(
                status=status,
                toolUseId=tool_use["toolUseId"],
                content=[ToolResultContent(text=response.content)],
            )
        )
