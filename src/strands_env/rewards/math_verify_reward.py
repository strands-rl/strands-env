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

r"""Reward function for math problems using math-verify for symbolic equivalence.

Uses HuggingFace's math-verify library to parse LaTeX/expressions from both
the model's `\\boxed{}` output and the ground truth, then checks equivalence
via SymPy (handling fractions, sets, nested expressions, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult
from strands_env.utils.decorators import with_timeout

logger = logging.getLogger(__name__)

# Suppress math_verify's noisy "Timeout during parsing" warnings
logging.getLogger("math_verify").setLevel(logging.ERROR)

# Use both LaTeX and expression extraction (math-verify's recommended default).
# LatexExtractionConfig handles \boxed{}, $...$, etc.
# ExprExtractionConfig handles plain "4", "0.5", "-3", etc.
_EXTRACTION_CONFIG = (LatexExtractionConfig(), ExprExtractionConfig())


class MathVerifyReward(RewardFunction):
    r"""Reward 1.0 if the model's `\boxed{}` answer is mathematically equivalent to ground truth.

    Args:
        float_rounding: Decimal places for float comparison (default 6).
        parse_timeout: Max seconds for parsing expressions from text (default 5).
        verify_timeout: Max seconds for SymPy simplification per comparison (default 5).
        answer_tail_chars: Only parse the last N chars of model response (default 500).
            Set to 0 to parse full response. The final `\\boxed{}` answer is typically
            at the end, so this avoids parsing long chain-of-thought reasoning.

    Notes:
        - Uses `math_verify.parse` to extract and convert answers to SymPy,
          then `math_verify.verify` for equivalence checking. This handles
          fraction/decimal equivalence, symbolic simplification, sets/intervals,
          and nested expressions with LaTeX normalization.
        - When either side parses to multiple candidate expressions (e.g. several
          `\\boxed{}` in the response), `verify` returns True if **any**
          gold-target pair matches (Cartesian product).
        - Uses `with_timeout` decorator for thread-safe timeouts. Math-verify's
          built-in timeout uses signal.alarm() which only works in the main thread.
          See: https://github.com/huggingface/Math-Verify/issues/42
    """

    def __init__(
        self,
        float_rounding: int = 6,
        parse_timeout: int = 5,
        verify_timeout: int = 5,
        answer_tail_chars: int = 500,
    ) -> None:
        """Initialize a `MathVerifyReward` instance."""
        self.float_rounding = float_rounding
        self.parse_timeout = parse_timeout
        self.verify_timeout = verify_timeout
        self.answer_tail_chars = answer_tail_chars

    def parse_expression(self, text: str) -> list:
        """Parse text into math expressions with timeout."""

        @with_timeout(self.parse_timeout)
        def _parse() -> list[Any]:
            return list(
                parse(
                    text,
                    extraction_config=_EXTRACTION_CONFIG,
                    parsing_timeout=None,  # Disable math-verify's timeout, use manual timeout
                    raise_on_error=True,
                )
            )

        return _parse()

    def verify_equivalence(self, gold: list, answer: list) -> Any:
        """Verify mathematical equivalence between gold and answer expressions with timeout."""

        @with_timeout(self.verify_timeout)
        def _verify() -> Any:
            return verify(
                gold=gold,
                target=answer,
                float_rounding=self.float_rounding,
                timeout_seconds=None,  # Disable math-verify's timeout, use manual timeout
            )

        return _verify()

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        ground_truth = action.task_context.ground_truth
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return RewardResult(reward=0.0, info={"reason": "invalid_ground_truth"})

        content = step_result.observation.final_response
        if content is None:
            return RewardResult(reward=0.0, info={"reason": "no_final_response"})

        # Parse ground truth
        try:
            gold = self.parse_expression(ground_truth)
            if not gold:
                raise ValueError("No ground truth expressions parsed")
        except Exception as e:
            logger.error("Failed to parse ground truth: %s: %s", type(e).__name__, ground_truth[:100])
            return RewardResult(reward=0.0, info={"reason": "gold_parse_failed", "ground_truth": ground_truth})

        # Parse model answer (only tail to avoid parsing long chain-of-thought)
        answer_text = content[-self.answer_tail_chars :] if self.answer_tail_chars else content
        try:
            answer = self.parse_expression(answer_text)
            if not answer:
                raise ValueError("No model answer expressions parsed")
        except Exception as e:
            logger.error("Failed to parse answer: %s: %s...", type(e).__name__, answer_text[:100])
            return RewardResult(reward=0.0, info={"reason": "answer_parse_failed", "response": content})

        # Verify equivalence
        try:
            matched = self.verify_equivalence(gold, answer)
        except Exception as e:
            logger.error("Failed to verify: %s", type(e).__name__)
            matched = False

        return RewardResult(
            reward=1.0 if matched else 0.0,
            info={
                "matched": matched,
                "gold_parsed": [str(g) for g in gold],
                "answer_parsed": [str(a) for a in answer],
                "ground_truth": ground_truth,
            },
        )
