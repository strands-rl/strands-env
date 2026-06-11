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
`tests/test.sh`) in an isolated Docker or EKS container. The agent gets a single
`execute_command` tool; `HarborReward` runs `tests/test.sh` for a binary reward.
Terminal-Bench and SWE-bench tasks share this exact contract, so both run on this
environment — they differ only in their dataset and system prompt (set per-benchmark).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as _HarborEnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.paths import TrialPaths
from harbor_aws.adapter import AWSEnvironment
from strands import tool
from typing_extensions import NotRequired, TypedDict, Unpack, override

from strands_env.core import Environment, ModelFactory
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RewardFunction

from .reward import HarborReward

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

    HarborEnvironment: TypeAlias = BaseEnvironment

HarborEnvironmentConfig: TypeAlias = _HarborEnvironmentConfig


class HarborConfig(EnvironmentConfig):
    """Serializable configuration for `HarborEnv`.

    Backends:
        - "docker": Local Docker via `harbor`'s native `DockerEnvironment`.
        - "eks": AWS EKS/Fargate via `harbor-aws`'s `AWSEnvironment`.
        - "e2b": Self-hosted e2b sandbox (Firecracker microVM, e2b-on-AWS) via
            `E2bAWSEnvironment`. Connection (`domain`/`api_key`) and template
            config go in `e2b_backend_config`, or fall back to the
            `E2B_DOMAIN` / `E2B_API_KEY` env vars.
    """

    task_id: str
    task_dir: str
    trial_dir: str
    timeout: NotRequired[int]
    backend: NotRequired[Literal["docker", "eks", "e2b"]]
    harbor_env_config: NotRequired[HarborEnvironmentConfig]
    eks_backend_config: NotRequired[EKSBackendConfig]
    e2b_backend_config: NotRequired[E2bBackendConfig]


class EKSBackendConfig(TypedDict, total=False):
    """Configuration for the EKS backend (harbor-aws)."""

    stack_name: str
    region: str
    ecr_cache: bool
    role_arn: str | None


class E2bBackendConfig(TypedDict, total=False):
    """Configuration for the e2b backend.

    Every field is optional and serializable, so a run is fully reproducible
    from config alone (mirrors `EKSBackendConfig`). `domain`/`api_key`, when
    provided, are written back into `E2B_DOMAIN`/`E2B_API_KEY` before the
    sandbox is built, because the underlying harbor `E2BEnvironment` + e2b SDK
    read those env vars directly. When omitted, the existing process env vars
    are used as-is.
    """

    # e2b cluster API domain. Falls back to the `E2B_DOMAIN` env var when unset.
    domain: str
    # e2b API key. Falls back to the `E2B_API_KEY` env var when unset. NOTE:
    # supplying this via config serializes it into the run config/logs; prefer
    # `api_key_file` or the env var if that's a concern.
    api_key: str
    # Path to a file containing the e2b API key (read + stripped at reset time).
    # Keeps the secret out of --env-config / config.json / shell history. Used
    # only when `api_key` is not set. `~` is expanded.
    api_key_file: str
    # Template id for the task. Optional: when omitted, the adapter resolves
    # the id from `templates_json` (or `E2B_TEMPLATES_PATH`) using the task
    # name as the lookup key.
    template_id: str
    # Path to a templates.json {task_name: template_id} mapping. Falls back to
    # the `E2B_TEMPLATES_PATH` env var when unset. Required (via either source)
    # unless `template_id` is provided directly.
    templates_json: str


class HarborEnv(Environment):
    """Harbor task environment using Harbor for container management and test execution."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[HarborConfig],
    ):
        """Initialize a `HarborEnv` instance."""
        super().__init__(model_factory=model_factory, reward_fn=None, **config)  # type: ignore[misc]
        self.task_id: str = str(self.config["task_id"])
        self.task_paths = TaskPaths(Path(str(self.config["task_dir"])))
        self.trial_paths = TrialPaths(Path(str(self.config["trial_dir"])))
        self.timeout: int = int(self.config.get("timeout", 1200))
        self.backend: Literal["docker", "eks", "e2b"] = self.config.get("backend", "docker")
        self.harbor_env_config: HarborEnvironmentConfig = self.config.get(
            "harbor_env_config", HarborEnvironmentConfig()
        )
        self.eks_backend_config: EKSBackendConfig = self.config.get("eks_backend_config", {})
        self.e2b_backend_config: E2bBackendConfig = self.config.get("e2b_backend_config", {})
        self.docker_env: HarborEnvironment | AWSEnvironment | None = None
        self.reward_fn = reward_fn or HarborReward(self)

    @override
    async def reset(self) -> None:
        """Build and start the container environment."""
        self.trial_paths.mkdir()
        session_id = f"{self.task_id}-{uuid.uuid4().hex[:8]}"

        force_build = True
        match self.backend:
            case "docker":
                self.docker_env = EnvironmentFactory.create_environment(
                    type=EnvironmentType.DOCKER,
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.harbor_env_config,
                )
            case "eks":
                from ._harbor_aws import ensure_harbor_aws_session

                await ensure_harbor_aws_session()
                self.docker_env = AWSEnvironment(
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.harbor_env_config,
                    **self.eks_backend_config,
                )
            case "e2b":
                from ._e2b_aws import E2bAWSEnvironment, resolve_template_id

                if domain := self.e2b_backend_config.get("domain"):
                    os.environ["E2B_DOMAIN"] = domain
                if api_key := self.e2b_backend_config.get("api_key"):
                    os.environ["E2B_API_KEY"] = api_key
                elif api_key_file := self.e2b_backend_config.get("api_key_file"):
                    key = Path(api_key_file).expanduser().read_text().strip()
                    if not key:
                        raise ValueError(f"e2b api_key_file is empty: {api_key_file}")
                    os.environ["E2B_API_KEY"] = key

                template_id = self.e2b_backend_config.get("template_id") or resolve_template_id(
                    task_name=self.task_id,
                    template_map_path=self.e2b_backend_config.get("templates_json"),
                )
                # force_build is ignored by E2bAWSEnvironment; templates are static.
                force_build = False
                self.docker_env = E2bAWSEnvironment(
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.harbor_env_config,
                    template_id=template_id,
                )

        await self.docker_env.start(force_build=force_build)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command in the environment.

        Args:
            command: The shell command to execute (e.g., "ls -la", "cat file.txt")

        Returns:
            Command output (stdout + stderr combined).
        """
        # TODO: Align the terminal command ouput with OpenHand's output format.
        if not self.docker_env:
            raise RuntimeError("Docker environment not initialized")
        result = await self.docker_env.exec(command, timeout_sec=self.timeout)
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
