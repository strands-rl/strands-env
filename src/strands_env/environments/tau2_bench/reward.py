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

"""Reward for tau2-bench: product of sub-rewards selected by `tau2_task.reward_basis`."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from strands.models import Model
from strands.types.content import Message
from typing_extensions import override

from strands_env.core.types import RewardFunction, RewardResult, RolloutResult, Task

from . import _tau2
from .nl_judge import Tau2BenchNLAssertionReward
from .simulator import Tau2BenchTerminationReason

if TYPE_CHECKING:
    from .env import Tau2BenchEnv

logger = logging.getLogger(__name__)


class Tau2BenchReward(RewardFunction):
    """Final reward = product of sub-rewards selected by `tau2_task.reward_basis`."""

    def __init__(self, env: Tau2BenchEnv, judge_model: Model | list[Model] | None = None) -> None:
        """Initialize a `Tau2BenchReward` instance."""
        self._env = env
        self._nl_judge = Tau2BenchNLAssertionReward(env, judge_model) if judge_model is not None else None

    @override
    async def compute(self, task: Task, result: RolloutResult) -> RewardResult:
        # tau2 only scores cleanly-stopped episodes, and only the user ends a dual-mode
        # dialogue — any other end (e.g. `max_steps`) scores 0.
        termination = self._env.user_simulator.termination
        if termination is not Tau2BenchTerminationReason.USER_STOP:
            return RewardResult(reward=0.0, info={"note": f"premature termination: {termination.value}"})

        reward_type = _tau2.RewardType
        basis_raw = self._env.tau2_task.evaluation_criteria.reward_basis
        basis = set(basis_raw) if basis_raw is not None else {reward_type.DB, reward_type.COMMUNICATE}
        messages = list(task.context.conversation_history) + list(result.messages)

        sub_rewards: dict[str, float] = {}
        nl_judge_info: dict[str, Any] | None = None
        if reward_type.DB in basis:
            sub_rewards["db"] = self._db_reward()
        if reward_type.ENV_ASSERTION in basis:
            sub_rewards["env_assertion"] = self._env_assertion_reward()
        if reward_type.ACTION in basis:
            sub_rewards["action"] = self._action_reward(messages)
        if reward_type.COMMUNICATE in basis:
            sub_rewards["communicate"] = self._communicate_reward(messages)
        if reward_type.NL_ASSERTION in basis:
            sub_rewards["nl_assertion"], nl_judge_info = await self._nl_assertion_reward(task, result)

        info: dict[str, Any] = {
            "sub_rewards": sub_rewards,
            "reward_basis": [b.value for b in basis],
        }
        if nl_judge_info is not None:
            info["nl_judge"] = nl_judge_info
        return RewardResult(reward=math.prod(sub_rewards.values()), info=info)

    def _db_reward(self) -> float:
        """Return 1.0 iff agent+user DB hashes match a golden env built by replaying golden actions on a fresh DB."""
        env, tau2_task = self._env, self._env.tau2_task
        gold = _tau2.build_task_environment(env.domain, tau2_task)
        for act in tau2_task.evaluation_criteria.actions or []:
            try:
                gold.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
            except Exception as e:
                logger.warning("Error in golden action %s(%s): %s", act.name, act.arguments, e)
        return float(
            gold.get_db_hash() == env.tau2_env.get_db_hash()
            and gold.get_user_db_hash() == env.tau2_env.get_user_db_hash()
        )

    def _env_assertion_reward(self) -> float:
        """Return 1.0 iff every `env_assertions` holds against the live post-episode env (telecom only)."""
        return float(
            all(
                self._env.tau2_env.run_env_assertion(a, raise_assertion_error=False)
                for a in self._env.tau2_task.evaluation_criteria.env_assertions or []
            )
        )

    def _action_reward(self, messages: list[Message]) -> float:
        """Return 1.0 iff every golden action is matched by some tool_use across agent + user-sim messages."""
        golden = self._env.tau2_task.evaluation_criteria.actions or []
        if not golden:
            return 1.0
        all_messages = messages + list(self._env.user_simulator.agent.messages)
        tool_calls = [
            _tau2.ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"], arguments=b["toolUse"]["input"])
            for m in all_messages
            if m.get("role") == "assistant"
            for b in m.get("content", [])
            if isinstance(b, dict) and "toolUse" in b
        ]
        return float(all(any(g.compare_with_tool_call(tc) for tc in tool_calls) for g in golden))

    def _communicate_reward(self, messages: list[Message]) -> float:
        """Return 1.0 iff each required info string appears in some assistant message (per-message search)."""
        required = self._env.tau2_task.evaluation_criteria.communicate_info or []
        if not required:
            return 1.0
        assistant_texts = [
            "".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
            for m in messages
            if m.get("role") == "assistant"
        ]
        # Per-message lower + comma-strip, matching tau2; each required info must match SOME message.
        haystacks = [t.lower().replace(",", "") for t in assistant_texts if t]
        return float(all(any(s.lower() in h for h in haystacks) for s in required))

    async def _nl_assertion_reward(self, task: Task, result: RolloutResult) -> tuple[float, dict[str, Any] | None]:
        """Return the NL_ASSERTION sub-reward and the judge's `info` (None when not judged).

        Defaults to 1.0 with no `info` when there are no assertions or no judge_model.
        """
        if not (self._env.tau2_task.evaluation_criteria.nl_assertions or []):
            return 1.0, None
        if self._nl_judge is None:
            logger.warning("NL_ASSERTION required but no judge_model; defaulting to 1.0")
            return 1.0, None
        nl_result = await self._nl_judge.compute(task, result)
        return nl_result.reward, nl_result.info
