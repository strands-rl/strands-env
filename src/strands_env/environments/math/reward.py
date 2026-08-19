from __future__ import annotations

import logging
from typing import Any, override

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

from strands_env.core.types import RewardFunction, RewardResult, RolloutResult, Task
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

    `math_verify.parse` converts both sides to SymPy and `math_verify.verify` compares them, which
    is what handles fraction/decimal equivalence, symbolic simplification, sets and intervals, and
    LaTeX normalization. When either side parses to several candidates — several `\boxed{}` in one
    response — a match on any gold-target pair counts.

    Notes:
        Timeouts go through `with_timeout` rather than math-verify's own, which uses
        `signal.alarm()` and so only fires on the main thread.
        See https://github.com/huggingface/Math-Verify/issues/42.
    """

    def __init__(
        self,
        float_rounding: int = 6,
        parse_timeout: int = 5,
        verify_timeout: int = 5,
        answer_tail_chars: int = 500,
    ) -> None:
        r"""Initialize a `MathVerifyReward` instance.

        Args:
            float_rounding: Decimal places for float comparison.
            parse_timeout: Max seconds to parse expressions from text.
            verify_timeout: Max seconds for SymPy simplification per comparison.
            answer_tail_chars: Parse only the last N chars of the response (0 = full text) —
                the final `\boxed{}` answer sits at the end, so this skips long chain-of-thought.
        """
        self.float_rounding = float_rounding
        self.parse_timeout = parse_timeout
        self.verify_timeout = verify_timeout
        self.answer_tail_chars = answer_tail_chars

    def parse_expression(self, text: str) -> list:
        """Parse `text` into candidate expressions, bounded by `parse_timeout`."""

        @with_timeout(self.parse_timeout)
        def _parse() -> list[Any]:
            return list(
                parse(
                    text,
                    extraction_config=_EXTRACTION_CONFIG,
                    # Annotated `int` upstream, but None is the documented way to opt out.
                    parsing_timeout=None,  # type: ignore[arg-type]
                    raise_on_error=True,
                )
            )

        return _parse()

    def verify_equivalence(self, gold: list, answer: list) -> Any:
        """Check gold/answer equivalence, bounded by `verify_timeout`."""

        @with_timeout(self.verify_timeout)
        def _verify() -> Any:
            return verify(
                gold=gold,
                target=answer,
                float_rounding=self.float_rounding,
                timeout_seconds=None,  # see `parse_expression`
            )

        return _verify()

    @override
    async def compute(self, task: Task, result: RolloutResult) -> RewardResult:
        """Score `result.final_response` against `task.ground_truth` by symbolic equivalence."""
        ground_truth = task.ground_truth
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return RewardResult(reward=0.0, info={"reason": "invalid_ground_truth"})

        content = result.final_response
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
