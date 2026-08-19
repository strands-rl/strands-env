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

    `reset()` fetches the container's tool list and keeps only what the task's `enabled_tools`
    names — a strict filter, so an empty one enables nothing. `cleanup()` drops that list and
    leaves the HTTP client alone.

    Notes:
        The `httpx.AsyncClient` is caller-owned: create it once (see `create_client`), pass it to
        every env, and close it after all tasks. Nothing here closes it.
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
            model_factory: builds the agent's model.
            http_client: shared client for the container; caller-owned.
            reward_fn: `None` means inference-only, no scoring.
            cached_tools: a pre-fetched `/list-tools` response, which skips the fetch in `reset()`.
                Share one across envs so the list isn't refetched per episode.
            **config: see `MCPAtlasConfig`.
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

        Caller-owned: close it with `async with` or an explicit `aclose()` once every task is done.
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
        """Clear the tool list; the shared HTTP client stays open (caller-owned)."""
        self._tools = []
