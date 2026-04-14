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

"""Reward function for MCP-Atlas benchmark.

Per-claim LLM-as-judge evaluation following MCP-Atlas's scoring methodology
(`mcp_evals_scores.py`).  Each GTFA claim is evaluated individually against
the agent's response using structured output, then scores are averaged.

Scoring (from MCP-Atlas):
    - `fulfilled` = 1.0
    - `partially_fulfilled` = 0.5
    - `not_fulfilled` = 0.0
    - `coverage_score` = mean across claims
    - pass threshold = 0.75
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from typing_extensions import override

from strands_env.core.types import Action, Observation, RewardResult, StepResult
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

logger = logging.getLogger(__name__)

# Pass threshold — task passes if coverage_score >= this value.
PASS_THRESHOLD = 0.75

# Scoring map — matches MCP-Atlas `CoverageEvaluator.evaluate()`.
COVERAGE_SCORES = {
    "fulfilled": 1.0,
    "partially_fulfilled": 0.5,
    "not_fulfilled": 0.0,
}

# Prompt template — verbatim from MCP-Atlas `_get_single_claim_evaluation_prompt()`.
CLAIM_EVALUATION_PROMPT = """\
You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same \
information as the claim, not on exact numerical precision unless precision is critical to \
the claim's meaning."""


class ClaimJudgment(BaseModel):
    """Structured output schema — matches MCP-Atlas `get_single_claim_evaluation_schema()`."""

    claim_text: str
    coverage_outcome: str = Field(pattern=r"^(fulfilled|partially_fulfilled|not_fulfilled)$")
    justification: str
    confidence_level: float = Field(ge=0.0, le=1.0)


class MCPAtlasRewardFunction(LLMJudgeReward[ClaimJudgment]):
    """Per-claim LLM-as-judge reward for MCP-Atlas benchmark.

    Overrides ``compute()`` to evaluate multiple claims per step via
    ``super().compute()`` and aggregate into a binary reward based on
    the 0.75 pass threshold.
    """

    judgment_format = ClaimJudgment

    @override
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Format the prompt for the current claim being evaluated."""
        return CLAIM_EVALUATION_PROMPT.format(claim=self._current_claim, response=self._response)

    @override
    async def get_reward(self, judgment: ClaimJudgment | str) -> float:
        """Convert a single claim judgment to a score."""
        if isinstance(judgment, str):
            return 0.0
        return COVERAGE_SCORES.get(judgment.coverage_outcome, 0.0)

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Evaluate each GTFA claim individually and return binary pass/fail reward."""
        claims: list[str] = action.task_context.gtfa_claims  # type: ignore[attr-defined]

        if not claims:
            return RewardResult(reward=self.default_reward, info={"reason": "no_claims"})

        self._response = Observation.get_final_response(step_result.observation.messages) or ""

        claim_results = []
        for claim in claims:
            self._current_claim = claim
            result = await super().compute(action, step_result)
            if result.info.get("status") == "error":
                return result
            judgment = result.info["judgment"]
            claim_results.append(
                {
                    "claim": claim,
                    "coverage_outcome": judgment["coverage_outcome"],
                    "score": result.reward,
                    "justification": judgment["justification"],
                    "confidence": judgment["confidence_level"],
                }
            )

        coverage_score = round(sum(r["score"] for r in claim_results) / len(claim_results), 3)
        reward = 1.0 if coverage_score >= PASS_THRESHOLD else 0.0

        return RewardResult(
            reward=reward,
            info={
                "status": "success",
                "coverage_score": coverage_score,
                "total_claims": len(claims),
                "fulfilled": sum(1 for r in claim_results if r["coverage_outcome"] == "fulfilled"),
                "partially_fulfilled": sum(1 for r in claim_results if r["coverage_outcome"] == "partially_fulfilled"),
                "not_fulfilled": sum(1 for r in claim_results if r["coverage_outcome"] == "not_fulfilled"),
                "claim_results": claim_results,
            },
        )
