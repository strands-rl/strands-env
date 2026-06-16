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

"""Harbor task environment for container management and test execution.

Runs any Harbor-format task (a directory with `task.toml`, `environment/Dockerfile`,
`tests/test.sh`) in an isolated Docker container or self-hosted e2b sandbox. The agent gets a single
`execute_command` tool; `HarborReward` runs `tests/test.sh` for a binary reward.
Terminal-Bench and SWE-bench tasks share this exact contract, so both run on this
environment — they differ only in their dataset and system prompt (set per-benchmark).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.paths import TrialPaths
from strands import tool
from typing_extensions import NotRequired, TypedDict, Unpack, override

from strands_env.core import Environment, ModelFactory
from strands_env.core.environment import EnvironmentConfig

from .reward import HarborReward

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

    HarborEnvironment: TypeAlias = BaseEnvironment


class HarborConfig(EnvironmentConfig):
    """Serializable configuration for `HarborEnv`.

    Backends:
        - "docker": Local Docker via `harbor`'s native `DockerEnvironment`.
        - "e2b": Self-hosted e2b sandbox (Firecracker microVM, e2b-on-AWS) via
            `PrebakedE2BEnvironment`. Connection (`domain`/`api_key`) and template
            config go in `prebaked_e2b_config`, or fall back to the
            `E2B_DOMAIN` / `E2B_API_KEY` env vars.
    """

    task_id: str
    task_dir: str
    trial_dir: str
    timeout: NotRequired[int]
    backend: NotRequired[Literal["docker", "e2b"]]
    task_env_config: NotRequired[TaskEnvironmentConfig]
    prebaked_e2b_config: NotRequired[PrebakedE2BConfig]


class PrebakedE2BConfig(TypedDict, total=False):
    """Connection + template config for the e2b backend (all fields optional)."""

    # e2b cluster API domain (env: E2B_DOMAIN).
    domain: str
    # e2b API key (env: E2B_API_KEY); prefer `api_key_file` to keep it out of config.
    api_key: str
    # Read the e2b API key from this file instead of inlining it.
    api_key_file: str
    # Template to boot; falls back to a `templates_json` lookup by task name.
    template_id: str
    # {task_name: template_id} JSON map (env: E2B_TEMPLATES_PATH).
    templates_json: str


class HarborEnv(Environment):
    """Harbor task environment using Harbor for container management and test execution."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        **config: Unpack[HarborConfig],
    ):
        """Initialize a `HarborEnv` instance."""
        super().__init__(model_factory=model_factory, **config)  # type: ignore[misc]
        self.task_id: str = str(self.config["task_id"])
        self.task_paths = TaskPaths(Path(str(self.config["task_dir"])))
        self.trial_paths = TrialPaths(Path(str(self.config["trial_dir"])))
        self.timeout: int = int(self.config.get("timeout", 1200))
        self.backend: Literal["docker", "e2b"] = self.config.get("backend", "docker")
        self.task_env_config: TaskEnvironmentConfig = self.config.get("task_env_config", TaskEnvironmentConfig())
        self.prebaked_e2b_config: PrebakedE2BConfig = self.config.get("prebaked_e2b_config", {})
        self.sandbox: HarborEnvironment | None = None
        # Harbor's reward is tied to the sandbox, so we don't need to pass it in.
        self.reward_fn = HarborReward(self)

    @override
    async def reset(self) -> None:
        """Build and start the sandbox."""
        self.trial_paths.mkdir()
        session_id = f"{self.task_id}-{uuid.uuid4().hex[:8]}"

        force_build = True
        match self.backend:
            case "docker":
                self.sandbox = EnvironmentFactory.create_environment(
                    type=EnvironmentType.DOCKER,
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.task_env_config,
                )
            case "e2b":
                from .e2b import PrebakedE2BEnvironment

                # we use prebaked e2b for self-hosting on e.g., AWS
                self.sandbox = PrebakedE2BEnvironment(
                    task_id=self.task_id,
                    prebaked_e2b_config=self.prebaked_e2b_config,
                    # below are the same as harbor's E2BEnvironment
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.task_env_config,
                )
                # Prebaked templates are static; force_build is a no-op here.
                force_build = False

        await self.sandbox.start(force_build=force_build)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command in the environment.

        Args:
            command: The shell command to execute (e.g., "ls -la", "cat file.txt")

        Returns:
            Command output (stdout + stderr combined).
        """
        # TODO: Align the terminal command ouput with OpenHand's output format.
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized")
        result = await self.sandbox.exec(command, timeout_sec=self.timeout)
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
        """Stop and delete the sandbox."""
        if self.sandbox:
            await self.sandbox.stop(delete=True)
            self.sandbox = None
