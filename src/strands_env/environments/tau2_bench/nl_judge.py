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

"""LLM judge for tau2's NL_ASSERTION sub-reward, aligned with tau2's judge prompt and schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from strands.models import Model
from typing_extensions import override

from strands_env.core.llm_judge_reward import LLMJudgeReward
from strands_env.core.types import RolloutResult, Task

if TYPE_CHECKING:
    from .env import Tau2BenchEnv

#: tau2's judge system prompt, dedented, with one deliberate deviation: snake_case keys
#: (the judge is a measurement instrument, not benchmark material — our judge model differs
#: from tau2's anyway). Field order follows tau2's own example block: verdict comes last,
#: conditioned on the written reasoning; upstream's bullet list contradicted its example.
NL_JUDGE_SYSTEM_PROMPT = """
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `expected_outcome`: repeat the expectation from the input that you are grading
- `reasoning`: a short explanation for your classification
- `met_expectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise

Example response structure:
{
    "results": [
        {
            "expected_outcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "met_expectation": <false or true>,
        }
    ]
}
""".strip()


class NLAssertion(BaseModel):
    """One NL-assertion judgment; field meanings live in `NL_JUDGE_SYSTEM_PROMPT`."""

    expected_outcome: str
    reasoning: str
    met_expectation: bool


class NLJudgment(BaseModel):
    """Mirrors tau2's expected JSON wrapper: ``{"results": [...]}``."""

    results: list[NLAssertion]


class Tau2BenchNLAssertionReward(LLMJudgeReward[NLJudgment]):
    """LLM-judged NL_ASSERTION sub-reward following tau2's judge design (see prompt note)."""

    judgment_format = NLJudgment

    def __init__(self, env: Tau2BenchEnv, judge_model: Model | list[Model]) -> None:
        """Initialize a `Tau2BenchNLAssertionReward` instance."""
        super().__init__(judge_model=judge_model, system_prompt=NL_JUDGE_SYSTEM_PROMPT)
        self._env = env

    @override
    async def get_judge_prompt(self, task: Task, result: RolloutResult) -> str:
        assertions = list(self._env.tau2_task.evaluation_criteria.nl_assertions or [])
        messages = list(task.conversation_history) + list(result.messages)
        lines = []
        for m in messages:
            for b in m.get("content") or []:
                if "text" in b:
                    lines.append(f"{m['role']}: {b['text']}")
                elif "toolResult" in b:
                    result_text = "".join(t.get("text", "") for t in b["toolResult"].get("content", []))
                    lines.append(f"tool: {result_text}")
        dialogue = "\n".join(lines)
        return f"conversation:\n{dialogue}\n\nexpected_outcomes:\n{assertions}"

    @override
    async def get_reward(self, judgment: NLJudgment | str) -> float:
        if not isinstance(judgment, NLJudgment):
            return self.default_reward  # 0.0 by default
        return float(all(r.met_expectation for r in judgment.results))
