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

"""AgentWorldModel environment backed by a FastAPI + SQLite server subprocess."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import httpx
from awm.tools import get_random_available_port
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands.types.tools import ToolResultContent
from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.tools.mcp_tool import MCPToolAdapter

from .reward import AWMRewardFunction
from .server import kill_server, wait_for_server, write_server_script

logger = logging.getLogger(__name__)

CLEANUP_TIMEOUT = 5
HTTP_MAX_CONNECTIONS = 1024
HTTP_MAX_KEEPALIVE_CONNECTIONS = 128
HTTP_CONNECT_TIMEOUT = 30.0
HTTP_READ_TIMEOUT = 600.0


class AWMMCPTool(MCPToolAdapter):
    """MCP tool backed by a `ClientSession` (single-server, direct connection).

    If `server_proc` is provided, checks `returncode` before each call
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
        """Initialize a `AWMMCPTool` instance."""
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


@dataclass
class AWMConfig:
    """Per-task configuration for `AWMEnvironment`.

    Attributes:
        scenario: Scenario name.
        envs_path: Path to gen_envs.jsonl.
        work_db_path: Working DB copy the server writes to.
        initial_db_path: Read-only DB snapshot for reward verification.
        temp_dir: Temp directory for server artifacts.
    """

    scenario: str
    envs_path: str
    work_db_path: str
    initial_db_path: str
    temp_dir: str


class AWMEnvironment(Environment):
    """MCP environment backed by an AWM FastAPI server subprocess.

    Notes:
        - `reset()` starts a per-task FastAPI server, opens an MCP session,
          and discovers tools.
        - `cleanup()` closes the session, kills the server, and removes the
          temp directory.
        - All instances share a single `httpx.AsyncClient` connection pool.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    _http_client: httpx.AsyncClient | None = None

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        config: AWMConfig,
        reward_fn: RewardFunction | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=120),
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        max_parallel_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        """Initialize an `AWMEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn or AWMRewardFunction(),
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            max_parallel_tool_calls=max_parallel_tool_calls,
            verbose=verbose,
        )
        self.config = config
        self._tool_call_timeout = tool_call_timeout
        self._server_proc: subprocess.Popen | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._tools: list[AWMMCPTool] = []

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if AWMEnvironment._http_client is None or AWMEnvironment._http_client.is_closed:
            AWMEnvironment._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=HTTP_MAX_CONNECTIONS,
                    max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
                ),
                timeout=httpx.Timeout(HTTP_CONNECT_TIMEOUT, read=HTTP_READ_TIMEOUT),
                follow_redirects=True,
            )
        return AWMEnvironment._http_client

    @override
    async def reset(self) -> None:
        """Start AWM server, open MCP session, discover tools."""
        port = get_random_available_port()
        script = Path(self.config.temp_dir) / "server.py"
        await asyncio.to_thread(
            write_server_script,
            script,
            port,
            self.config.scenario,
            self.config.envs_path,
            self.config.work_db_path,
        )

        self._server_proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        await wait_for_server(port, self.config.scenario)
        logger.info(
            "AWM server pid=%d for %s on port %d",
            self._server_proc.pid,
            self.config.scenario,
            port,
        )

        # Open MCP session and discover tools
        stack = contextlib.AsyncExitStack()
        try:
            transport = streamable_http_client(
                f"http://localhost:{port}/mcp",
                http_client=self._get_http_client(),
                terminate_on_close=False,
            )
            read_stream, write_stream, *_ = await stack.enter_async_context(transport)
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            result = await session.list_tools()
            self._tools = [
                AWMMCPTool(
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
        await kill_server(self._server_proc, CLEANUP_TIMEOUT)
        self._server_proc = None
        if self.config.temp_dir:
            await asyncio.to_thread(shutil.rmtree, self.config.temp_dir, True)
