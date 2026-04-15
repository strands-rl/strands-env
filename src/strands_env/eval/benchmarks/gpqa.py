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

"""Evaluator for GPQA (Graduate-Level Google-Proof Q&A) benchmarks."""

from __future__ import annotations

import logging
import random
import re
import sys
import unicodedata
from collections.abc import Iterable

from datasets import load_dataset
from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.core.types import RewardFunction, RewardResult, StepResult

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


class GPQAReward(RewardFunction):
    """Rule-based reward for GPQA multiple-choice benchmarks.

    Answer extraction follows the `multi_choice_regex` filter from
    `lm-evaluation-harness` with a 3-level fallback:

    1. Last parenthesised letter `(A)`-`(D)` in the response.
    2. Choice **text** found verbatim in the response (case/punct insensitive).
    3. Bare letter after a colon, e.g. `Answer: B`.
    """

    #: Primary regex — matches `(A)` through `(Z)` with parentheses (case-sensitive,
    #: matching lm-evaluation-harness where ``ignore_case`` only applies to choice text).
    PRIMARY_PATTERN: re.Pattern[str] = re.compile(r"\(([A-Z])\)")

    #: Fallback regex — bare letter after a colon, e.g. `Answer: B`.
    COLON_PATTERN: re.Pattern[str] = re.compile(r":[\s]*([A-D])")

    #: Punctuation translation table matching lm-evaluation-harness.
    PUNCT_TBL: dict[int, None] = dict.fromkeys(
        i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
    )

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Extract the answer letter and compare with the correct choice."""
        response = step_result.observation.final_response or ""
        correct_letter: str | None = getattr(action.task_context, "correct_letter", None)

        if correct_letter is None:
            return RewardResult(reward=0.0, info={"status": "error", "error": "missing correct_letter in TaskContext"})

        choices: list[str] | None = getattr(action.task_context, "choices", None)
        extracted = self.extract_answer(response, choices)
        is_correct = extracted is not None and extracted.upper() == correct_letter.upper()

        return RewardResult(
            reward=1.0 if is_correct else 0.0,
            info={
                "status": "success",
                "extracted_answer": extracted,
                "correct_letter": correct_letter,
            },
        )

    @classmethod
    def extract_answer(cls, text: str, choices: list[str] | None = None) -> str | None:
        """Extract the answer letter (A-D) from the model's response.

        Uses the same 3-level fallback as `lm-evaluation-harness` `multi_choice_regex`:

        1. Last parenthesised letter `(A)`-`(D)` in the response.
        2. Choice text found verbatim in the response (case + punctuation insensitive).
        3. Bare letter `A`-`D` immediately after a colon.
        """
        # Level 1: last (A)-(Z) in text (comparison against correct_letter handles A-D filtering)
        matches = cls.PRIMARY_PATTERN.findall(text)
        if matches:
            return matches[-1]

        # Level 2: match choice text in response (case/punct insensitive)
        if choices:
            normalised_resp = text.lower().translate(cls.PUNCT_TBL)
            for i, choice in enumerate(choices):
                normalised_choice = choice.lower().translate(cls.PUNCT_TBL)
                if normalised_choice and normalised_choice in normalised_resp:
                    return chr(65 + i)  # A, B, C, D

        # Level 3: bare letter after colon
        colon_matches = cls.COLON_PATTERN.findall(text)
        if colon_matches:
            return colon_matches[-1].upper()

        return None


class GPQAEvaluator(Evaluator):
    """Base evaluator for GPQA benchmarks.

    Loads the `Idavidrein/gpqa` dataset (gated; requires HuggingFace login and
    dataset access). Choice text is cleaned with the same ``preprocess``
    function used by lm-evaluation-harness, and choices are shuffled with a
    per-sample deterministic seed for reproducibility.
    """

    benchmark_name: str = "gpqa"
    dataset_path: str = "Idavidrein/gpqa"

    CHOICE_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

    @staticmethod
    def preprocess(text: str | None) -> str:
        """Clean choice text, matching lm-evaluation-harness ``process_docs``."""
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load GPQA dataset from HuggingFace (streaming).

        Answer choices are preprocessed (matching lm-evaluation-harness) and
        shuffled with a per-sample deterministic seed for reproducibility. Each
        action message contains the question followed by four labeled answer
        choices (A)-(D).

        Yields:
            Action objects with formatted multiple-choice question and ground truth.
        """
        try:
            dataset = load_dataset(
                self.dataset_path, self.benchmark_name.replace("-", "_"), split="train", streaming=True
            )
        except Exception as e:
            if "gated" in str(e).lower():
                raise PermissionError(
                    f"{self.dataset_path} is a gated dataset. "
                    "Accept the license at https://huggingface.co/datasets/Idavidrein/gpqa "
                    "then authenticate via `huggingface-cli login` or set the HF_TOKEN env var."
                ) from e
            raise

        for i, row in enumerate(dataset):
            question = row.get("Question")
            correct_answer = row.get("Correct Answer")
            incorrect_1 = row.get("Incorrect Answer 1")
            incorrect_2 = row.get("Incorrect Answer 2")
            incorrect_3 = row.get("Incorrect Answer 3")

            if not all((question, correct_answer, incorrect_1, incorrect_2, incorrect_3)):
                logger.warning("Row %s: missing required fields, skipped", i)
                continue

            # Preprocess choice text (matching lm-evaluation-harness)
            choices = [
                self.preprocess(incorrect_1),
                self.preprocess(incorrect_2),
                self.preprocess(incorrect_3),
                self.preprocess(correct_answer),
            ]
            correct_preprocessed = self.preprocess(correct_answer)

            # Shuffle choices deterministically per sample
            rng = random.Random(i)
            rng.shuffle(choices)
            correct_idx = choices.index(correct_preprocessed)
            correct_letter = self.CHOICE_LETTERS[correct_idx]

            # Format question with labeled choices
            choice_lines = "\n".join(
                f"({letter}) {choice}" for letter, choice in zip(self.CHOICE_LETTERS, choices, strict=True)
            )
            formatted_question = (
                f"What is the correct answer to this question:\n{question}\n\n"
                f"Choices:\n{choice_lines}\n\n"
                f'Format your response as follows: "The correct answer is (insert answer here)"'
            )

            yield Action(
                message=formatted_question,
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{i}",
                    ground_truth=str(correct_answer),
                    **{
                        "subdomain": row.get("Subdomain", ""),
                        "correct_letter": correct_letter,
                        "choices": choices,
                    },
                ),
            )


@register_eval("gpqa-diamond")
class GPQADiamondEvaluator(GPQAEvaluator):
    """GPQA Diamond subset (198 expert-validated questions)."""

    benchmark_name = "gpqa-diamond"


@register_eval("gpqa-main")
class GPQAMainEvaluator(GPQAEvaluator):
    """GPQA Main subset (448 questions)."""

    benchmark_name = "gpqa-main"


@register_eval("gpqa-experts")
class GPQAExpertsEvaluator(GPQAEvaluator):
    """GPQA Experts subset (expert-only validated questions)."""

    benchmark_name = "gpqa-experts"


@register_eval("gpqa-extended")
class GPQAExtendedEvaluator(GPQAEvaluator):
    """GPQA Extended subset (546 questions)."""

    benchmark_name = "gpqa-extended"
