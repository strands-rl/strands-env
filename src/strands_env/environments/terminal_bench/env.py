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

"""Terminal-Bench environment using Harbor's DockerEnvironment for container management and test execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.paths import TrialPaths
from strands import tool
from typing_extensions import override

from strands_env.core import Environment, ModelFactory
from strands_env.core.types import RewardFunction

from .reward import TerminalBenchRewardFunction

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment


@dataclass
class TerminalBenchConfig:
    """Configuration for task-dependent arguments in `TerminalBenchEnv`.

    Attributes:
        task_id: Unique identifier for the task.
        task_dir: Path to task directory containing Dockerfile, tests/, environment/.
        trial_dir: Path to trial output directory for storing results.
        env_config: Harbor EnvironmentConfig for Docker setup.
        timeout_s: Timeout in seconds for test execution.
    """

    task_id: str
    task_dir: Path
    trial_dir: Path
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    timeout_s: int = 1200


class TerminalBenchEnv(Environment):
    """Terminal-Bench environment using Harbor's DockerEnvironment for container management and test execution."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        config: TerminalBenchConfig,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = 25,
        max_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        """Initialize the Terminal-Bench environment with a model factory and task configuration."""
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=None,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )

        self.config = config
        self.task_paths = TaskPaths(config.task_dir)
        self.trial_paths = TrialPaths(trial_dir=config.trial_dir)
        self.docker_env: BaseEnvironment | None = None
        self.reward_fn = reward_fn or TerminalBenchRewardFunction(self)

    @override
    async def reset(self) -> None:
        """Build and start the Docker environment."""
        self.trial_paths.mkdir()
        session_id = f"{self.config.task_id}-{uuid.uuid4().hex[:8]}"
        self.docker_env = EnvironmentFactory.create_environment(
            type=EnvironmentType.DOCKER,
            environment_dir=self.task_paths.environment_dir,
            environment_name=session_id,
            session_id=session_id,
            trial_paths=self.trial_paths,
            task_env_config=self.config.env_config,
        )
        await self.docker_env.start(force_build=True)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command in the environment.

        Args:
            command: The shell command to execute (e.g., "ls -la", "cat file.txt")

        Returns:
            Command output (stdout + stderr combined).
        """
        if not self.docker_env:
            raise RuntimeError("Docker environment not initialized")
        result = await self.docker_env.exec(command, timeout_sec=self.config.timeout_s)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.return_code != 0:
            output += f"\n[exit code]: {result.return_code}"
        return output.strip() or "(no output)"

    @override
    def get_tools(self) -> list:
        """Return the execute_command tool."""
        return [self.execute_command]

    @override
    async def cleanup(self) -> None:
        """Stop and delete the Docker environment."""
        if self.docker_env:
            await self.docker_env.stop(delete=True)
            self.docker_env = None
