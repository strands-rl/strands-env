# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP environment for connecting an agent to an MCP server via `MCPClient`."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands.tools.mcp import MCPClient
from typing_extensions import override

from strands_env.core.environment import Environment

if TYPE_CHECKING:
    from strands.tools.mcp import MCPAgentTool

logger = logging.getLogger(__name__)


class MCPEnvironment(Environment):
    """Environment backed by a single MCP server.

    Accepts an optional pre-constructed `MCPClient` and manages its lifecycle:
    `reset()` starts the client, `cleanup()` stops it.
    `get_tools()` returns tools from the client.

    Subclasses may set `self._mcp_client` during `reset()` and call `super().reset()`
    to start it.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        mcp_client: MCPClient | None = None,
        **kwargs: Any,
    ):
        """Initialize the MCP environment with an optional pre-constructed MCP client."""
        super().__init__(**kwargs)
        self._mcp_client = mcp_client

    @override
    async def reset(self) -> None:
        if self._mcp_client:
            await asyncio.to_thread(self._mcp_client.start)

    @override
    def get_tools(self) -> list[MCPAgentTool]:
        """Return tools from the MCP client."""
        if self._mcp_client is None:
            return []
        return list(self._mcp_client.list_tools_sync())

    @override
    async def cleanup(self) -> None:
        if self._mcp_client:
            await asyncio.to_thread(self._mcp_client.stop, None, None, None)
            self._mcp_client = None
