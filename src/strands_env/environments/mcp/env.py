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

"""MCP environment base class for connecting an agent to MCP servers."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction

from .tool import MCPTool

logger = logging.getLogger(__name__)

TOOL_CALL_TIMEOUT = timedelta(seconds=120)


class MCPEnvironment(Environment):
    """Base environment backed by MCP servers.

    Notes:
        - Provides shared tool storage (`self._tools`) and a default
          `get_tools()` / `cleanup()` implementation.
        - Subclasses override `reset()` to set up their connection, populate
          `self._tools`, and manage their own connection lifecycle.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        tool_call_timeout: timedelta = TOOL_CALL_TIMEOUT,
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        max_parallel_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        """Initialize a `MCPEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=reward_fn,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            max_parallel_tool_calls=max_parallel_tool_calls,
            verbose=verbose,
        )
        self._tool_call_timeout = tool_call_timeout
        self._tools: list[MCPTool] = []

    @override
    def get_tools(self) -> list:
        """Return tool list directly."""
        return list(self._tools)

    @override
    async def cleanup(self) -> None:
        """Clear tool list. Subclasses should close their own connections first, then call super."""
        self._tools = []
