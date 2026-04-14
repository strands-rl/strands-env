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

"""Code sandbox environment using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from typing_extensions import Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.tools import CodeInterpreterQuotas, CodeInterpreterToolkit
from strands_env.utils.aws import get_client

if TYPE_CHECKING:
    from strands_env.core.models import ModelFactory
    from strands_env.core.types import RewardFunction
    from strands_env.utils.aws import BotoClient


class CodeSandboxConfig(EnvironmentConfig, total=False):
    """Serializable configuration for `CodeSandboxEnv`."""

    mode: Literal["code", "terminal", "code_and_terminal"]


class CodeSandboxEnv(Environment):
    """Code sandbox environment using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        Provides `execute_code` (Python) and/or `execute_command` (shell) tools
        depending on the configured mode.
    """

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        client: BotoClient | None = None,
        quotas: CodeInterpreterQuotas | None = None,
        **config: Unpack[CodeSandboxConfig],
    ):
        """Initialize a `CodeSandboxEnv` instance."""
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)  # type: ignore[misc]
        self.mode: str = self.config.get("mode", "code")
        self._toolkit = CodeInterpreterToolkit(
            client=client or get_client(service_name="bedrock-agentcore"), session_name="strands-env", quotas=quotas
        )

    @override
    def get_tools(self) -> list:
        """Return tools based on configured mode."""
        match self.mode:
            case "code":
                return [self._toolkit.execute_code]
            case "terminal":
                return [self._toolkit.execute_command]
            case "code_and_terminal":
                return [self._toolkit.execute_code, self._toolkit.execute_command]
            case _:
                raise ValueError(f"Invalid mode: {self.mode}")

    @override
    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        await self._toolkit.cleanup()
