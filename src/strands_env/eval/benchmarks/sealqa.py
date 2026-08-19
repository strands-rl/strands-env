from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import override

from datasets import load_dataset

from strands_env.core import Task

from ..evaluator import Evaluator
from ..registry import register_eval
from .simpleqa_verified import SimpleQAReward

logger = logging.getLogger(__name__)

# SealQA reward is the same as SimpleQA reward
SealQAReward = SimpleQAReward

# ---------------------------------------------------------------------------
# Evaluators — Seal-0 and Seal-Hard
# ---------------------------------------------------------------------------


class SealQAEvaluator(Evaluator):
    """Base evaluator for SealQA benchmarks."""

    hf_dataset_path = "vtllms/sealqa"

    @override
    def load_dataset(self) -> Iterable[Task]:
        """Load SealQA dataset from HuggingFace.

        Yields:
            Task objects with question text, ground truth, and task metadata.
        """
        dataset = load_dataset(self.hf_dataset_path, name=self.hf_dataset_config, split="test", streaming=True)

        for i, row in enumerate(dataset):
            question, answer = row.get("question"), row.get("answer")
            if question is None or answer is None:
                logger.warning("Row %s: missing question/answer, skipped", i)
                continue

            yield Task(
                id=f"{self.benchmark_name}_{i}",
                message=str(question),
                ground_truth=str(answer),
                # untyped metadata bag, kept in saved results for analysis
                **{
                    k: row.get(k)
                    for k in ("freshness", "question_types", "effective_year", "search_results", "topic", "urls")
                },
            )


@register_eval("sealqa-seal-0")
class Seal0Evaluator(SealQAEvaluator):
    """SealQA Seal-0 benchmark (111 core questions)."""

    hf_dataset_config = "seal_0"


@register_eval("sealqa-seal-hard")
class SealHardEvaluator(SealQAEvaluator):
    """SealQA Seal-Hard benchmark (254 difficult questions)."""

    hf_dataset_config = "seal_hard"
