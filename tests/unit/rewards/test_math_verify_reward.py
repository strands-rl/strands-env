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

"""Tests for MathVerifyReward."""

from strands_env.core.types import Action, Observation, StepResult, TaskContext
from strands_env.rewards.math_verify_reward import MathVerifyReward


def make_step_result(content: str) -> StepResult:
    msg = {"role": "assistant", "content": [{"text": content}]}
    return StepResult(observation=Observation(messages=[msg]))


def make_action(ground_truth: str | None = None) -> Action:
    return Action(message="test", task_context=TaskContext(ground_truth=ground_truth))


class TestMathRewardFunction:
    """Core behavior of MathVerifyReward."""

    async def test_exact_match(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("4"), make_step_result("The answer is $\\boxed{4}$."))
        assert result.reward == 1.0
        assert result.info["matched"] is True

    async def test_wrong_answer(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("4"), make_step_result("The answer is $\\boxed{5}$."))
        assert result.reward == 0.0
        assert result.info["matched"] is False

    async def test_fraction_decimal_equivalence(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("0.5"), make_step_result("$\\boxed{\\frac{1}{2}}$"))
        assert result.reward == 1.0

    async def test_symbolic_equivalence(self):
        fn = MathVerifyReward()
        result = await fn.compute(
            make_action("$\\frac{\\sqrt{3}}{3}$"),
            make_step_result("$\\boxed{\\frac{1}{\\sqrt{3}}}$"),
        )
        assert result.reward == 1.0

    async def test_negative_number(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("-3"), make_step_result("$\\boxed{-3}$"))
        assert result.reward == 1.0

    async def test_latex_fraction_in_gold(self):
        fn = MathVerifyReward()
        result = await fn.compute(
            make_action("\\frac{7}{12}"),
            make_step_result("$\\boxed{\\frac{7}{12}}$"),
        )
        assert result.reward == 1.0


class TestMultipleBoxed:
    """Handling of multiple \\boxed{} in the response."""

    async def test_any_boxed_matches(self):
        """verify() uses Cartesian product: any match counts."""
        fn = MathVerifyReward()
        result = await fn.compute(
            make_action("42"),
            make_step_result("First attempt $\\boxed{99}$, correction $\\boxed{42}$."),
        )
        assert result.reward == 1.0

    async def test_none_match(self):
        fn = MathVerifyReward()
        result = await fn.compute(
            make_action("42"),
            make_step_result("$\\boxed{1}$ and $\\boxed{2}$"),
        )
        assert result.reward == 0.0


class TestEdgeCases:
    """Edge cases and error paths."""

    async def test_no_final_response(self):
        fn = MathVerifyReward()
        step = StepResult(observation=Observation(messages=[{"role": "user", "content": [{"text": "hi"}]}]))
        result = await fn.compute(make_action("4"), step)
        assert result.reward == 0.0
        assert result.info["reason"] == "no_final_response"

    async def test_invalid_ground_truth_none(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action(None), make_step_result("$\\boxed{4}$"))
        assert result.reward == 0.0
        assert result.info["reason"] == "invalid_ground_truth"

    async def test_invalid_ground_truth_empty(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action(""), make_step_result("$\\boxed{4}$"))
        assert result.reward == 0.0
        assert result.info["reason"] == "invalid_ground_truth"

    async def test_no_boxed_but_expr_extracted(self):
        """ExprExtractionConfig picks up bare numbers in prose, so this still matches."""
        fn = MathVerifyReward()
        result = await fn.compute(make_action("4"), make_step_result("The answer is 4."))
        assert result.reward == 1.0

    async def test_no_parseable_answer(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("4"), make_step_result("I don't know the answer."))
        assert result.reward == 0.0
        assert result.info["reason"] == "answer_parse_failed"

    async def test_unparseable_gold(self):
        """Ground truth that math-verify can't parse."""
        fn = MathVerifyReward()
        result = await fn.compute(
            make_action("not a math expression at all!"),
            make_step_result("$\\boxed{4}$"),
        )
        assert result.reward == 0.0
        assert result.info["reason"] == "gold_parse_failed"

    async def test_info_contains_parsed_representations(self):
        fn = MathVerifyReward()
        result = await fn.compute(make_action("4"), make_step_result("$\\boxed{4}$"))
        assert "gold_parsed" in result.info
        assert "answer_parsed" in result.info
        assert "ground_truth" in result.info
