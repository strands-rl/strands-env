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

"""Reward function for the Harbor task environment."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task.task import Task as HarborTaskSpec
from harbor.models.trial.paths import EnvironmentPaths
from harbor.verifier.verifier import Verifier

from strands_env.core.types import RewardFunction, RewardResult, RolloutResult

if TYPE_CHECKING:
    from .env import HarborEnv
    from .task import HarborTask

logger = logging.getLogger(__name__)


class HarborReward(RewardFunction["HarborTask"]):
    """Run harbor's Verifier in the sandbox and return its reward."""

    def __init__(self, env: HarborEnv) -> None:
        """Initialize a `HarborReward` instance."""
        self._env = env

    async def compute(self, task: HarborTask, result: RolloutResult) -> RewardResult:
        """Run harbor's Verifier in the sandbox and return its reward."""
        try:
            assert self._env.sandbox is not None, "sandbox not initialized"
            timeout = task.verifier_timeout if task.verifier_timeout is not None else self._env.exec_timeout
            await self._env.sandbox.exec(f"mkdir -p {EnvironmentPaths.verifier_dir}")
            verifier = Verifier(
                task=HarborTaskSpec(Path(task.task_dir)),
                trial_paths=task.trial_paths,
                environment=self._env.sandbox,
            )
            verifier_result = await asyncio.wait_for(verifier.verify(), timeout=timeout)
            # reward.json is unvalidated upstream; loud indexing raises on a missing key, never a silent 0
            return RewardResult(reward=float(verifier_result.rewards["reward"]), info={"status": "success"})
        except Exception as e:
            logger.exception("Verification failed due to %s: %s", type(e).__name__, str(e))
            return RewardResult(reward=0.0, info={"status": "error", "message": str(e)})
