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

"""MCP-Atlas environment backed by a Docker container."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, NotRequired, Unpack, override

import httpx
from mcp.types import Tool as MCPToolDef

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction

from .task import MCPAtlasTask
from .tool import MCPAtlasTool

logger = logging.getLogger(__name__)


class MCPAtlasConfig(EnvironmentConfig):
    """Serializable configuration for `MCPAtlasEnv`."""

    tool_timeout: NotRequired[int]


class MCPAtlasEnv(Environment[MCPAtlasTask]):
    """MCP-Atlas benchmark environment backed by a Docker container.

    Notes:
        - A shared `httpx.AsyncClient` is passed in at construction time;
          the caller owns its lifecycle (create once, close after all tasks).
        - `reset()` fetches tools from the container and applies the task's
          `enabled_tools` filter strictly (an empty filter enables no tools).
        - `cleanup()` clears the tool list only.
    """

    DEFAULT_DOCKER_URL: ClassVar[str] = "http://localhost:1984"

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        http_client: httpx.AsyncClient,
        reward_fn: RewardFunction[MCPAtlasTask] | None = None,
        cached_tools: list[dict] | None = None,
        **config: Unpack[MCPAtlasConfig],
    ):
        """Initialize an `MCPAtlasEnv` instance.

        Args:
            model_factory: Factory for the agent's model.
            http_client: Shared client for the MCP-Atlas container (see `create_client`);
                the caller owns its lifecycle.
            reward_fn: Optional reward function (None = inference-only).
            cached_tools: Pre-fetched `/list-tools` response; skips the fetch in `reset()`.
                Share one across envs to avoid refetching per episode.
            **config: See `MCPAtlasConfig`.
        """
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn,
            **config,  # type: ignore[misc]
        )
        self._http_client = http_client
        self._cached_tools = cached_tools
        self._tools: list[MCPAtlasTool] = []
        self._tool_timeout: int = int(self.config.get("tool_timeout", 60))

    @staticmethod
    def create_client(
        base_url: str = DEFAULT_DOCKER_URL,
        *,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ) -> httpx.AsyncClient:
        """Create an `httpx.AsyncClient` configured for the MCP-Atlas container.

        The caller owns the returned client's lifecycle and should close it
        when done (e.g. via `async with` or explicit `aclose()`).

        Args:
            base_url: Base URL of the MCP-Atlas Docker container.
            max_connections: Maximum number of concurrent connections.
            max_keepalive_connections: Maximum number of idle keep-alive connections.
        """
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
        )
        return httpx.AsyncClient(base_url=base_url, limits=limits)

    @override
    async def reset(self, task: MCPAtlasTask) -> None:
        """Fetch tools from the container (or use cache) and apply the task's tool filter."""
        if self._cached_tools is None:
            response = await self._http_client.post("/list-tools", timeout=self._tool_timeout)
            response.raise_for_status()
            self._cached_tools = response.json()
        enabled_tools = set(task.enabled_tools)
        self._tools = [
            MCPAtlasTool(MCPToolDef.model_validate(tool), self._http_client, timeout=self._tool_timeout)
            for tool in self._cached_tools
            if tool["name"] in enabled_tools
        ]
        logger.info("MCP-Atlas: %d tools enabled", len(self._tools))

    @override
    def get_tools(self) -> list:
        """Return the MCP tools discovered during `reset()`."""
        return list(self._tools)

    @override
    async def cleanup(self) -> None:
        """Clear tool list. The shared HTTP client is not closed here."""
        self._tools = []
