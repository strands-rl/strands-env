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
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import NotRequired, Unpack, override

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory

from .reward import AgentWorldModelReward
from .server import kill_server, start_server
from .task import AgentWorldModelTask
from .tool import AgentWorldModelTool

logger = logging.getLogger(__name__)


class AgentWorldModelConfig(EnvironmentConfig):
    """Serializable configuration for `AgentWorldModelEnv` (per-task fields ride `AgentWorldModelTask`)."""

    envs_path: str  # path to gen_envs.jsonl — the dataset of generated worlds
    mcp_timeout: NotRequired[int]  # seconds per MCP tool call (default 60)


class AgentWorldModelEnv(Environment[AgentWorldModelTask]):
    """MCP environment backed by an AgentWorldModel FastAPI server subprocess.

    Notes:
        - `reset(task)` clones the task's DB snapshot into fresh scratch, starts a
          per-episode FastAPI server on the clone, opens an MCP session, and
          discovers tools.
        - `cleanup()` closes the session, kills the server, and removes the scratch dir.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        http_client: httpx.AsyncClient | None = None,
        **config: Unpack[AgentWorldModelConfig],
    ):
        """Initialize an `AgentWorldModelEnv` instance.

        Args:
            model_factory: Factory for the agent's model.
            http_client: Optional shared HTTP client for the MCP transport.
            **config: See `AgentWorldModelConfig`.
        """
        super().__init__(model_factory=model_factory, **config)  # type: ignore[misc]
        self.envs_path = Path(str(self.config["envs_path"]))
        self.mcp_timeout = timedelta(seconds=int(self.config.get("mcp_timeout", 60)))
        self.temp_dir: Path | None = None
        self._http_client = http_client
        self._server_proc: subprocess.Popen | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[AgentWorldModelTool] = []
        # The reward is tied to the env's working DB, so we don't take it as a parameter.
        self.reward_fn = AgentWorldModelReward(self)

    @property
    def work_db_path(self) -> Path | None:
        """The episode's working DB (a private clone of the task's snapshot); None outside an episode."""
        return self.temp_dir / "work.db" if self.temp_dir is not None else None

    @override
    async def reset(self, task: AgentWorldModelTask) -> None:
        """Clone the task's DB into fresh scratch, start the server, open an MCP session."""
        # Per-episode scratch: the working DB is a private copy of the task's pristine
        # snapshot, so concurrent episodes (e.g. pass@k) never share mutable state.
        self.temp_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="awm-"))
        work_db_path = self.temp_dir / "work.db"
        await asyncio.to_thread(shutil.copyfile, task.initial_db_path, work_db_path)

        await asyncio.sleep(random.uniform(0, 5))  # stagger concurrent server spawns
        self._server_proc, port = await start_server(
            task.scenario,
            self.envs_path,
            work_db_path,
            self.temp_dir,
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
                AgentWorldModelTool(
                    tool,
                    session,
                    server_proc=self._server_proc,
                    timeout=self.mcp_timeout,
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
        if self.temp_dir is not None:
            await asyncio.to_thread(shutil.rmtree, self.temp_dir, True)
            self.temp_dir = None
