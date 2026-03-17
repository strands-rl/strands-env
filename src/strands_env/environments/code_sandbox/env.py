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

"""Code sandbox environment using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.tools import CodeInterpreterQuotas, CodeInterpreterToolkit
from strands_env.utils.aws import get_client

if TYPE_CHECKING:
    from botocore.client import BaseClient

    from strands_env.core.models import ModelFactory
    from strands_env.core.types import RewardFunction


class CodeMode(str, Enum):
    """Tool modes for `CodeSandboxEnv`."""

    CODE = "code"
    """Only `execute_code` tool (Python execution)."""

    TERMINAL = "terminal"
    """Only `execute_command` tool (shell commands)."""

    CODE_AND_TERMINAL = "code_and_terminal"
    """Both `execute_code` and `execute_command` tools."""


class CodeSandboxEnv(Environment):
    """Code sandbox environment using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        Provides `execute_code` (Python) and/or `execute_command` (shell) tools
        depending on the configured `CodeMode`.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = 10,
        max_tool_calls: int | None = 50,
        verbose: bool = False,
        client: BaseClient | None = None,
        mode: CodeMode = CodeMode.CODE,
        quotas: CodeInterpreterQuotas | None = None,
    ):
        """Initialize a `CodeSandboxEnv` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=system_prompt,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )
        self.mode = mode
        self._toolkit = CodeInterpreterToolkit(
            client=client or get_client(service_name="bedrock-agentcore"),
            quotas=quotas,
        )

    @override
    def get_tools(self) -> list:
        """Return tools based on configured mode."""
        tool_map: dict[CodeMode, list] = {
            CodeMode.CODE: [self._toolkit.execute_code],
            CodeMode.TERMINAL: [self._toolkit.execute_command],
            CodeMode.CODE_AND_TERMINAL: [self._toolkit.execute_code, self._toolkit.execute_command],
        }
        return tool_map[self.mode]

    @override
    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        await self._toolkit.cleanup()
