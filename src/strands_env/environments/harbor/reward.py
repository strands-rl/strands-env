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
        """Run verification tests in Docker and return a binary reward."""
        try:
            reward = await self._run_verification(task)
            return RewardResult(reward=reward, info={"status": "success"})
        except Exception as e:
            logger.exception("Verification failed due to %s: %s", type(e).__name__, str(e))
            return RewardResult(reward=0.0, info={"status": "error", "message": str(e)})

    async def _run_verification(self, task: HarborTask) -> float:
        """Run harbor's own Verifier and return its reward."""
        assert self._env.sandbox is not None, "Sandbox not initialized"
        sandbox = self._env.sandbox
        timeout = task.verifier_timeout if task.verifier_timeout is not None else self._env.exec_timeout

        # harbor exec is a non-login `bash -c` (no ~/.profile), so user-local tools — like the
        # uv that swebench images install into ~/.local/bin — are not on PATH. Expose them
        # additively; the graded swebench baseline (70.8%, #78) relied on an equivalent PATH
        # prepend in the pre-Verifier flow.
        await sandbox.exec('ln -sf "$HOME/.local/bin/"* /usr/local/bin/ 2>/dev/null || true')

        verifier = Verifier(
            task=HarborTaskSpec(Path(task.task_dir)),
            trial_paths=task.trial_paths,
            environment=sandbox,
        )
        result = await asyncio.wait_for(verifier.verify(), timeout=timeout)

        # Harbor's raw reward, as parsed from reward.json (first) or reward.txt. Current
        # datasets emit binary 0/1 by construction; partial-credit tasks pass through intact.
        return float(result.rewards.get("reward", 0.0))
