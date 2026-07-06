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

"""Reward function for AgentWorldModel tasks.

Executes per-task `verify_task_completion` via `exec()` for binary reward.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from strands_env.core.types import RewardFunction, RewardResult, RolloutResult

from .task import AgentWorldModelTask

if TYPE_CHECKING:
    from .env import AgentWorldModelEnv

logger = logging.getLogger(__name__)


class AgentWorldModelReward(RewardFunction[AgentWorldModelTask]):
    """Binary reward via execution-based verification."""

    def __init__(self, env: AgentWorldModelEnv) -> None:
        """Initialize an `AgentWorldModelReward` instance (the env owns the episode's working DB)."""
        self._env = env

    async def compute(self, task: AgentWorldModelTask, result: RolloutResult) -> RewardResult:
        """Run the task's verification code against the final DB state and the agent's final response."""
        work_db_path = self._env.work_db_path
        assert work_db_path is not None, "reset() has not run"

        # blocking exec + SQLite I/O, shipped to a thread below
        def _verify() -> dict:
            namespace: dict = {"sqlite3": sqlite3, "json": json}
            exec(task.verify_code, namespace)  # noqa: S102
            return namespace["verify_task_completion"](
                initial_db_path=task.initial_db_path,
                final_db_path=str(work_db_path),
                final_answer=result.final_response or "",
            )

        try:
            verification = await asyncio.to_thread(_verify)
        except Exception as e:
            logger.warning("Verification failed for %s task %s: %s", task.scenario, task.task_idx, e)
            return RewardResult(
                reward=0.0,
                info={"status": "error", "error": str(e), "error_type": type(e).__name__},
            )

        # status is "success" whenever the verifier RAN — reward 0.0 means the agent failed the
        # task, not the pipeline (the same contract as the other envs; evaluators retry on error).
        is_complete = isinstance(verification, dict) and verification.get("result") == "complete"
        logger.info("Verification %s task %d: %s", task.scenario, task.task_idx, verification)
        return RewardResult(
            reward=1.0 if is_complete else 0.0, info={"status": "success", "verification_result": verification}
        )
