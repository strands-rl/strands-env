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

from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Message
from typing_extensions import override

from strands_env.core.llm_judge_reward import LLMJudgeReward
from strands_env.core.types import RewardFunction, RewardResult, RolloutResult, Task

from . import _tau2
from .simulator import Tau2BenchTerminationReason

if TYPE_CHECKING:
    from ._tau2 import Tau2Task
    from .env import Tau2BenchEnv

logger = logging.getLogger(__name__)

#: Terminations tau2 treats as a clean stop (only the user ends a dual-mode dialogue);
#: any other end (e.g. `max_steps`) scores 0.
CLEAN_TERMINATIONS = frozenset({Tau2BenchTerminationReason.USER_STOP})


#: Verbatim system prompt from tau2's `evaluator_nl_assertions.py`.
NL_JUDGE_SYSTEM_PROMPT = """
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `reasoning`: a short explanation for your classification
- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
- `expectedOutcome`: repeat the expectation from the input that you are grading

Example response structure:
{
    "results": [
        {
            "expectedOutcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "metExpectation": <false or true>,
        }
    ]
}
""".strip()


class NLAssertion(BaseModel):
    """One NL-assertion judgment.

    Fields are camelCase (hence `noqa: N815`) to match the JSON keys tau2's judge prompt asks the
    model to emit; their meaning lives in `NL_JUDGE_SYSTEM_PROMPT`, not in per-field `Field` descriptions.
    """

    expectedOutcome: str  # noqa: N815
    reasoning: str
    metExpectation: bool  # noqa: N815


class NLJudgment(BaseModel):
    """Mirrors tau2's expected JSON wrapper: ``{"results": [...]}``."""

    results: list[NLAssertion]


class Tau2BenchNLAssertionReward(LLMJudgeReward[NLJudgment]):
    """LLM-judged NL_ASSERTION sub-reward; byte-aligned with tau2's prompt/schema."""

    judgment_format = NLJudgment

    def __init__(self, env: Tau2BenchEnv, judge_model: Model | list[Model]) -> None:
        """Initialize a `Tau2BenchNLAssertionReward` instance."""
        super().__init__(judge_model=judge_model, system_prompt=NL_JUDGE_SYSTEM_PROMPT)
        self._env = env

    @override
    async def get_judge_prompt(self, task: Task, result: RolloutResult) -> str:
        tau2_task = self._env.tau2_task
        assertions = list(tau2_task.evaluation_criteria.nl_assertions or [])
        messages = list(task.context.conversation_history) + list(result.messages)
        lines = []
        for m in messages:
            for b in m.get("content") or []:
                if "text" in b:
                    lines.append(f"{m['role']}: {b['text']}")
                elif "toolResult" in b:
                    result_text = "".join(t.get("text", "") for t in b["toolResult"].get("content", []))
                    lines.append(f"tool: {result_text}")
        dialogue = "\n".join(lines)
        return f"conversation:\n{dialogue}\n\nexpectedOutcomes:\n{assertions}"

    @override
    async def get_reward(self, judgment: NLJudgment | str) -> float:
        if not isinstance(judgment, NLJudgment):
            return 0.0
        return float(all(r.metExpectation for r in judgment.results))


class Tau2BenchReward(RewardFunction):
    """Final reward = product of sub-rewards selected by `tau2_task.reward_basis`."""

    def __init__(self, env: Tau2BenchEnv, judge_model: Model | list[Model] | None = None) -> None:
        """Initialize a `Tau2BenchReward` instance."""
        self._env = env
        self._nl_judge = Tau2BenchNLAssertionReward(env, judge_model) if judge_model is not None else None

    @override
    async def compute(self, task: Task, result: RolloutResult) -> RewardResult:
        # tau2 only scores cleanly-stopped episodes; a premature end (e.g. `max_steps`) scores 0.
        termination = self._env.user_simulator.termination if self._env.user_simulator is not None else None
        if termination not in CLEAN_TERMINATIONS:
            return RewardResult(reward=0.0, info={"note": f"premature termination: {termination}"})

        reward_type = _tau2.RewardType
        tau2_task = self._env.tau2_task
        basis_raw = tau2_task.evaluation_criteria.reward_basis
        basis = set(basis_raw) if basis_raw is not None else {reward_type.DB, reward_type.COMMUNICATE}
        messages = list(task.context.conversation_history) + list(result.messages)

        sub_rewards: dict[str, float] = {}
        nl_judge_info: dict[str, Any] | None = None
        if reward_type.DB in basis:
            sub_rewards["db"] = _db_reward(self._env, tau2_task)
        if reward_type.ENV_ASSERTION in basis:
            sub_rewards["env_assertion"] = _env_assertion_reward(self._env, tau2_task)
        if reward_type.ACTION in basis:
            sub_rewards["action"] = _action_reward(self._env, messages, tau2_task)
        if reward_type.COMMUNICATE in basis:
            sub_rewards["communicate"] = _communicate_reward(messages, tau2_task)
        if reward_type.NL_ASSERTION in basis:
            sub_rewards["nl_assertion"], nl_judge_info = await self._nl_assertion_reward(task, result, tau2_task)

        info: dict[str, Any] = {
            "sub_rewards": sub_rewards,
            "reward_basis": [b.value for b in basis],
        }
        if nl_judge_info is not None:
            info["nl_judge"] = nl_judge_info
        return RewardResult(reward=math.prod(sub_rewards.values()), info=info)

    async def _nl_assertion_reward(
        self, task: Task, result: RolloutResult, tau2_task: Tau2Task
    ) -> tuple[float, dict[str, Any] | None]:
        """Return the NL_ASSERTION sub-reward and the judge's `info` (None when not judged).

        Defaults to 1.0 with no `info` when there are no assertions or no judge_model.
        """
        if not (tau2_task.evaluation_criteria.nl_assertions or []):
            return 1.0, None
        if self._nl_judge is None:
            logger.warning("NL_ASSERTION required but no judge_model; defaulting to 1.0")
            return 1.0, None
        nl_result = await self._nl_judge.compute(task, result)
        return nl_result.reward, nl_result.info


def _db_reward(env: Tau2BenchEnv, tau2_task: Tau2Task) -> float:
    """Return 1.0 iff the agent+user DB hashes match a golden env built by replaying `tau2_task.actions` on a fresh DB."""
    gold = _tau2.build_task_environment(env.domain, tau2_task)
    for act in tau2_task.evaluation_criteria.actions or []:
        try:
            gold.make_tool_call(tool_name=act.name, requestor=act.requestor, **act.arguments)
        except Exception as e:
            logger.warning("Error in golden action %s(%s): %s", act.name, act.arguments, e)
    return float(
        gold.get_db_hash() == env.tau2_env.get_db_hash() and gold.get_user_db_hash() == env.tau2_env.get_user_db_hash()
    )


def _env_assertion_reward(env: Tau2BenchEnv, tau2_task: Tau2Task) -> float:
    """Return 1.0 iff every `tau2_task.env_assertions` holds against the live post-episode env (telecom only)."""
    return float(
        all(
            env.tau2_env.run_env_assertion(a, raise_assertion_error=False)
            for a in tau2_task.evaluation_criteria.env_assertions or []
        )
    )


def _action_reward(env: Tau2BenchEnv, messages: list[Message], tau2_task: Tau2Task) -> float:
    """Return 1.0 iff every golden action is matched by some tool_use across agent + user-sim messages."""
    golden = tau2_task.evaluation_criteria.actions or []
    if not golden:
        return 1.0
    all_messages = messages + (env.user_simulator.agent.messages if env.user_simulator is not None else [])
    tool_calls = [
        _tau2.ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"], arguments=b["toolUse"]["input"])
        for m in all_messages
        if m.get("role") == "assistant"
        for b in m.get("content", [])
        if isinstance(b, dict) and "toolUse" in b
    ]
    return float(all(any(g.compare_with_tool_call(tc) for tc in tool_calls) for g in golden))


def _communicate_reward(messages: list[Message], tau2_task: Tau2Task) -> float:
    """Return 1.0 iff each required info string appears in some assistant message (per-message search)."""
    required = tau2_task.evaluation_criteria.communicate_info or []
    if not required:
        return 1.0
    assistant_texts = [
        "".join(b.get("text", "") for b in m.get("content", []) if isinstance(b, dict))
        for m in messages
        if m.get("role") == "assistant"
    ]
    # Per-message lower + comma-strip; each required info must match SOME message.
    haystacks = [t.lower().replace(",", "") for t in assistant_texts if t]
    return float(all(any(s.lower() in h for h in haystacks) for s in required))
