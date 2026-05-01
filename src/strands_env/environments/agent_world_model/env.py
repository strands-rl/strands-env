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

"""AgentWorldModel environment backed by a FastAPI + SQLite server subprocess."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from typing_extensions import NotRequired, Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction

from .reward import AgentWorldModelRewardFunction
from .server import kill_server, start_server
from .tool import AgentWorldModelMCPTool

logger = logging.getLogger(__name__)


class AgentWorldModelConfig(EnvironmentConfig):
    """Serializable configuration for `AgentWorldModelEnvironment`."""

    scenario: str
    envs_path: str
    work_db_path: str
    initial_db_path: str
    temp_dir: str
    tool_call_timeout: NotRequired[int]


class AgentWorldModelEnvironment(Environment):
    """MCP environment backed by an AgentWorldModel FastAPI server subprocess.

    Notes:
        - `reset()` starts a per-task FastAPI server, opens an MCP session,
          and discovers tools.
        - `cleanup()` closes the session, kills the server, and removes the
          temp directory.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        http_client: httpx.AsyncClient | None = None,
        **config: Unpack[AgentWorldModelConfig],
    ):
        """Initialize an `AgentWorldModelEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn or AgentWorldModelRewardFunction(),
            **config,  # type: ignore[misc]
        )
        self._http_client = http_client
        self._tool_call_timeout = timedelta(seconds=int(self.config.get("tool_call_timeout", 60)))
        self._scenario: str = str(self.config["scenario"])
        self._envs_path = Path(str(self.config["envs_path"]))
        self._work_db_path = Path(str(self.config["work_db_path"]))
        self._temp_dir = Path(str(self.config["temp_dir"]))
        self._server_proc: subprocess.Popen | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[AgentWorldModelMCPTool] = []

    @override
    async def reset(self) -> None:
        """Start AgentWorldModel server, open MCP session, discover tools."""
        await asyncio.sleep(random.uniform(0, 5))  # stagger concurrent server spawns
        self._server_proc, port = await start_server(
            self._scenario,
            self._envs_path,
            self._work_db_path,
            self._temp_dir,
        )

        # Open MCP session and discover tools
        stack = contextlib.AsyncExitStack()
        try:
            transport = streamable_http_client(
                f"http://localhost:{port}/mcp",
                http_client=self._http_client,
                terminate_on_close=False,  # we kill the server ourselves in cleanup()
            )
            read_stream, write_stream, *_ = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            result = await session.list_tools()
            self._tools = [
                AgentWorldModelMCPTool(
                    tool,
                    session,
                    server_proc=self._server_proc,
                    timeout=self._tool_call_timeout,
                )
                for tool in result.tools
            ]
            logger.info("Listed %d MCP tools", len(self._tools))
        except BaseException:
            await stack.aclose()
            raise

        self._exit_stack = stack

    @override
    def get_tools(self) -> list:
        """Return the MCP tools discovered during `reset()`."""
        return list(self._tools)

    @override
    async def cleanup(self) -> None:
        """Close MCP session/transport, kill server, remove temp dir."""
        self._tools = []
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None
        await kill_server(self._server_proc)
        self._server_proc = None
        if self._temp_dir:
            await asyncio.to_thread(shutil.rmtree, self._temp_dir, True)
