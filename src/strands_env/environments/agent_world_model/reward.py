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

Executes per-task ``verify_task_completion`` via ``exec()`` for binary reward.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import traceback
from typing import Any

from strands_env.core.types import Action, Observation, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)

OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_AGENT_FAILED = "AGENT_FAILED"
OUTCOME_VERIFY_ERROR = "VERIFY_ERROR"


def _run_verification(verify_code: str, initial_db_path: str, work_db_path: str, final_answer: str) -> dict:
    """Execute verification code in a thread (blocking exec + SQLite I/O)."""
    namespace: dict = {"sqlite3": sqlite3, "json": json}
    exec(verify_code, namespace)  # noqa: S102
    return namespace["verify_task_completion"](
        initial_db_path=initial_db_path,
        final_db_path=work_db_path,
        final_answer=final_answer,
    )


class AgentWorldModelRewardFunction(RewardFunction):
    """Binary reward via execution-based verification."""

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run verification code against the agent's final response."""
        ctx: Any = action.task_context
        final_answer = Observation.get_final_response(step_result.observation.messages) or ""

        try:
            result = await asyncio.to_thread(
                _run_verification,
                ctx.verify_code,
                ctx.initial_db_path,
                ctx.work_db_path,
                final_answer,
            )
        except Exception as e:
            logger.warning("Verification failed for %s task %s: %s", ctx.scenario, ctx.task_idx, e)
            return RewardResult(
                reward=0.0,
                info={
                    "outcome": OUTCOME_VERIFY_ERROR,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exception_only(type(e), e)[-1].strip(),
                },
            )

        is_complete = isinstance(result, dict) and result.get("result") == "complete"
        outcome = OUTCOME_COMPLETED if is_complete else OUTCOME_AGENT_FAILED
        logger.info("Verification %s task %d: %s (outcome=%s)", ctx.scenario, ctx.task_idx, result, outcome)
        return RewardResult(
            reward=1.0 if is_complete else 0.0, info={"outcome": outcome, "verification_result": result}
        )
