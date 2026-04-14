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

"""Evaluator for HMMT (Harvard-MIT Mathematics Tournament) benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from datasets import load_dataset
from typing_extensions import override

from strands_env.core import Action, TaskContext

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


class HMMTEvaluator(Evaluator):
    """Base evaluator for HMMT math competition problems."""

    benchmark_name: str = "hmmt"
    dataset_path: str = ""

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load HMMT dataset from HuggingFace (streaming).

        Yields:
            Action objects with problem text and ground truth.
        """
        dataset = load_dataset(self.dataset_path, split="train", streaming=True)

        for i, row in enumerate(dataset):
            problem, answer = row.get("problem"), row.get("answer")
            if problem is None or answer is None:
                logger.warning("Row %s: missing problem/answer, skipped", i)
                continue
            yield Action(
                message=str(problem),
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{row.get('problem_idx', i)}",
                    ground_truth=str(answer),
                ),
            )


@register_eval("hmmt-feb-2025")
class HMMTFeb2025Evaluator(HMMTEvaluator):
    """HMMT February 2025 benchmark."""

    benchmark_name = "hmmt-feb-2025"
    dataset_path = "MathArena/hmmt_feb_2025"


@register_eval("hmmt-nov-2025")
class HMMTNov2025Evaluator(HMMTEvaluator):
    """HMMT November 2025 benchmark."""

    benchmark_name = "hmmt-nov-2025"
    dataset_path = "MathArena/hmmt_nov_2025"


@register_eval("hmmt-feb-2026")
class HMMTFeb2026Evaluator(HMMTEvaluator):
    """HMMT February 2026 benchmark."""

    benchmark_name = "hmmt-feb-2026"
    dataset_path = "MathArena/hmmt_feb_2026"
