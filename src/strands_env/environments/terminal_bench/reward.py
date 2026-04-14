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

"""Reward function for Terminal-Bench environment."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from harbor.models.trial.paths import EnvironmentPaths

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

if TYPE_CHECKING:
    from .env import TerminalBenchEnv

logger = logging.getLogger(__name__)


class TerminalBenchReward(RewardFunction):
    """Execute test scripts in Docker and compute binary reward (0 or 1)."""

    def __init__(self, env: TerminalBenchEnv) -> None:
        """Initialize a `TerminalBenchReward` instance."""
        self._env = env

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run verification tests in Docker and return a binary reward."""
        try:
            reward = await self._run_verification()
            return RewardResult(reward=reward)
        except Exception as e:
            logger.exception("Verification failed due to %s: %s", type(e).__name__, str(e))
            return RewardResult(reward=0.0, info={"error": str(e)})

    async def _run_verification(self) -> float:
        """Upload tests, execute `test.sh`, download results, and parse reward."""
        assert self._env.docker_env is not None, "Docker environment not initialized"
        docker_env = self._env.docker_env
        task_paths = self._env.task_paths
        trial_paths = self._env.trial_paths
        timeout = self._env.timeout

        # Upload and run tests.
        await docker_env.upload_dir(source_dir=task_paths.tests_dir, target_dir="/tests")
        test_cmd = (
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"bash /tests/test.sh 2>&1 | tee {EnvironmentPaths.verifier_dir}/test-stdout.txt"
        )
        await docker_env.exec(test_cmd, timeout_sec=timeout)

        # Download results if not using mounted volumes
        if not docker_env.is_mounted:
            await docker_env.download_dir(
                source_dir=str(EnvironmentPaths.verifier_dir),
                target_dir=trial_paths.verifier_dir,
            )

        # Parse reward (1.0 if reward.txt contains value >= 1, else 0.0)
        reward_path = trial_paths.reward_text_path
        if reward_path.exists() and reward_path.stat().st_size > 0:
            return 1.0 if float(reward_path.read_text().strip()) >= 1.0 else 0.0
        logger.warning("No reward file at %s", reward_path)
        return 0.0
