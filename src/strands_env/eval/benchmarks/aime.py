from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import override

from datasets import load_dataset

from strands_env.core import Task

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


class AIMEEvaluator(Evaluator):
    """Base evaluator for AIME math competition problems."""

    @override
    def load_dataset(self) -> Iterable[Task]:
        """Load AIME dataset from HuggingFace (streaming).

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
                id=f"{self.benchmark_name}_{row.get('id', i)}",
                message=str(problem),
                ground_truth=str(answer),
            )


@register_eval("aime-2024")
class AIME2024Evaluator(AIMEEvaluator):
    """AIME 2024 benchmark."""

    hf_dataset_path = "HuggingFaceH4/aime_2024"


@register_eval("aime-2025")
class AIME2025Evaluator(AIMEEvaluator):
    """AIME 2025 benchmark."""

    hf_dataset_path = "MathArena/aime_2025"


@register_eval("aime-2026")
class AIME2026Evaluator(AIMEEvaluator):
    """AIME 2026 benchmark."""

    hf_dataset_path = "MathArena/aime_2026"
