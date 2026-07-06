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
from typing import TYPE_CHECKING, Any, override

from strands.models import Model
from strands.types.content import Message

from strands_env.core.types import RewardFunction, RewardResult, RolloutResult

from . import _tau2
from .nl_judge import Tau2BenchNLAssertionReward
from .simulator import Tau2BenchTerminationReason
from .task import Tau2BenchTask

if TYPE_CHECKING:
    from .env import Tau2BenchEnv

logger = logging.getLogger(__name__)


class Tau2BenchReward(RewardFunction[Tau2BenchTask]):
    """Final reward = product of sub-rewards selected by `tau2_task.reward_basis`."""

    def __init__(self, env: Tau2BenchEnv, judge_model: Model | list[Model] | None = None) -> None:
        self._env = env
        self._nl_judge = Tau2BenchNLAssertionReward(env, judge_model) if judge_model is not None else None

    @override
    async def compute(self, task: Tau2BenchTask, result: RolloutResult) -> RewardResult:
        # tau2 only scores cleanly-stopped episodes with user stop
        termination = self._env.user_simulator.termination
        if termination is not Tau2BenchTerminationReason.USER_STOP:
            return RewardResult(reward=0.0, info={"note": f"premature termination: {termination.value}"})

        # full dialogue = seeded greeting exchange + everything the episode added
        messages = list(task.conversation_history) + list(result.messages)

        # compute sub-rewards
        reward_basis = set(task.tau2_task.evaluation_criteria.reward_basis)
        info: dict[str, Any] = {"reward_basis": sorted(b.value for b in reward_basis), "sub_rewards": {}}
        for reward_type in reward_basis:
            match reward_type:
                case _tau2.RewardType.DB:
                    info["sub_rewards"]["db"] = self._db_reward(task)
                case _tau2.RewardType.ENV_ASSERTION:
                    info["sub_rewards"]["env_assertion"] = self._env_assertion_reward(task)
                case _tau2.RewardType.ACTION:
                    info["sub_rewards"]["action"] = self._action_reward(task, messages)
                case _tau2.RewardType.COMMUNICATE:
                    info["sub_rewards"]["communicate"] = self._communicate_reward(task, messages)
                case _tau2.RewardType.NL_ASSERTION:
                    if self._nl_judge is None:
                        raise ValueError("task requires NL_ASSERTION but no judge_model is configured")
                    nl_result = await self._nl_judge.compute(task, result)
                    info["sub_rewards"]["nl_assertion"] = nl_result.reward
                    if nl_result.info:
                        info["nl_judge"] = nl_result.info

        return RewardResult(reward=math.prod(info["sub_rewards"].values()), info=info)

    def _db_reward(self, task: Tau2BenchTask) -> float:
        """Return 1.0 iff agent+user DB hashes match a golden env built by replaying golden actions on a fresh DB."""
        # `task.tau2_env` is the live world the episode mutated; the gold env is a fresh replay
        ref_env = _tau2.build_task_environment(task.domain, task.tau2_task)
        for act in task.tau2_task.evaluation_criteria.actions or []:
            try:
                ref_env.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
            except Exception as e:
                logger.warning("Error in golden action %s(%s): %s", act.name, act.arguments, e)
        # compare db hashes
        return float(
            ref_env.get_db_hash() == task.tau2_env.get_db_hash()
            and ref_env.get_user_db_hash() == task.tau2_env.get_user_db_hash()
        )

    def _env_assertion_reward(self, task: Tau2BenchTask) -> float:
        """Return 1.0 iff every `env_assertions` holds against the live post-episode env (telecom only)."""
        return float(
            all(
                task.tau2_env.run_env_assertion(a, raise_assertion_error=False)
                for a in task.tau2_task.evaluation_criteria.env_assertions or []
            )
        )

    def _action_reward(self, task: Tau2BenchTask, messages: list[Message]) -> float:
        """Return 1.0 iff every golden action is matched by some tool call across agent + user-sim messages."""
        ref_actions = task.tau2_task.evaluation_criteria.actions or []
        if not ref_actions:
            return 1.0
        # collect every toolUse block (assistant-role) from both sides of the dialogue as tau2 ToolCalls
        tool_calls = [
            _tau2.ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"], arguments=b["toolUse"]["input"])
            for m in messages + list(self._env.user_simulator.agent.messages)
            if m.get("role") == "assistant"
            for b in m.get("content", [])
            if isinstance(b, dict) and "toolUse" in b
        ]
        return float(all(any(g.compare_with_tool_call(tc) for tc in tool_calls) for g in ref_actions))

    def _communicate_reward(self, task: Tau2BenchTask, messages: list[Message]) -> float:
        """Return 1.0 iff each required info string appears in some assistant message (per-message search)."""
        required_texts = task.tau2_task.evaluation_criteria.communicate_info or []
        if not required_texts:
            return 1.0
        assistant_texts = [
            "".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
            for m in messages
            if m.get("role") == "assistant"
        ]
        # Per-message lower + comma-strip, matching tau2; each required info must match SOME message.
        haystacks = [t.lower().replace(",", "") for t in assistant_texts if t]
        return float(all(any(s.lower() in h for h in haystacks) for s in required_texts))
