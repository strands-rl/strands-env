from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import override

from datasets import load_dataset

from strands_env.core import Task

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


class HMMTEvaluator(Evaluator):
    """Base evaluator for HMMT math competition problems."""

    @override
    def load_dataset(self) -> Iterable[Task]:
        """Load HMMT dataset from HuggingFace (streaming).

        Yields:
            Task objects with problem text and ground truth.
        """
        dataset = load_dataset(self.hf_dataset_path, split="train", streaming=True)

        for i, row in enumerate(dataset):
            problem, answer = row.get("problem"), row.get("answer")
            if problem is None or answer is None:
                logger.warning("Row %s: missing problem/answer, skipped", i)
                continue
            yield Task(
                id=f"{self.benchmark_name}_{row.get('problem_idx', i)}",
                message=str(problem),
                ground_truth=str(answer),
            )


@register_eval("hmmt-feb-2025")
class HMMTFeb2025Evaluator(HMMTEvaluator):
    """HMMT February 2025 benchmark."""

    hf_dataset_path = "MathArena/hmmt_feb_2025"


@register_eval("hmmt-nov-2025")
class HMMTNov2025Evaluator(HMMTEvaluator):
    """HMMT November 2025 benchmark."""

    hf_dataset_path = "MathArena/hmmt_nov_2025"


@register_eval("hmmt-feb-2026")
class HMMTFeb2026Evaluator(HMMTEvaluator):
    """HMMT February 2026 benchmark."""

    hf_dataset_path = "MathArena/hmmt_feb_2026"
