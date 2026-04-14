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

"""Evaluator for SealQA (SEarch-Augmented Language models QA) benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from datasets import load_dataset
from typing_extensions import override

from strands_env.core import Action, TaskContext

from ..evaluator import EvalSample, Evaluator
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

    benchmark_name: str = "sealqa"
    dataset_path: str = "vtllms/sealqa"
    dataset_config: str = ""

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where the judge failed (e.g. throttling), so they are retried on resume."""
        reward = sample.step_result.reward
        if reward is None:
            return True
        return reward.info.get("status") != "error"

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load SealQA dataset from HuggingFace.

        Yields:
            Action objects with question text, ground truth, and task metadata.
        """
        dataset = load_dataset(self.dataset_path, name=self.dataset_config, split="test", streaming=True)

        for i, row in enumerate(dataset):
            question, answer = row.get("question"), row.get("answer")
            if question is None or answer is None:
                logger.warning("Row %s: missing question/answer, skipped", i)
                continue

            yield Action(
                message=str(question),
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{i}",
                    ground_truth=str(answer),
                    **{
                        "freshness": row.get("freshness"),
                        "question_types": row.get("question_types"),
                        "effective_year": row.get("effective_year"),
                        "search_results": row.get("search_results"),
                        "topic": row.get("topic"),
                        "urls": row.get("urls"),
                    },
                ),
            )


@register_eval("sealqa-seal-0")
class Seal0Evaluator(SealQAEvaluator):
    """SealQA Seal-0 benchmark (111 core questions)."""

    benchmark_name = "sealqa-seal-0"
    dataset_config = "seal_0"


@register_eval("sealqa-seal-hard")
class SealHardEvaluator(SealQAEvaluator):
    """SealQA Seal-Hard benchmark (254 difficult questions)."""

    benchmark_name = "sealqa-seal-hard"
    dataset_config = "seal_hard"
