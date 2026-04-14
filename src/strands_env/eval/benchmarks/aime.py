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

"""Evaluator for AIME (American Invitational Mathematics Examination) benchmarks."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from datasets import load_dataset
from typing_extensions import override

from strands_env.core import Action, TaskContext

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


class AIMEEvaluator(Evaluator):
    """Base evaluator for AIME math competition problems."""

    benchmark_name: str = "aime"
    dataset_path: str = ""

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load AIME dataset from HuggingFace (streaming).

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
                    id=f"{self.benchmark_name}_{row.get('id', i)}",
                    ground_truth=str(answer),
                ),
            )


@register_eval("aime-2024")
class AIME2024Evaluator(AIMEEvaluator):
    """AIME 2024 benchmark."""

    benchmark_name = "aime-2024"
    dataset_path = "HuggingFaceH4/aime_2024"


@register_eval("aime-2025")
class AIME2025Evaluator(AIMEEvaluator):
    """AIME 2025 benchmark."""

    benchmark_name = "aime-2025"
    dataset_path = "MathArena/aime_2025"


@register_eval("aime-2026")
class AIME2026Evaluator(AIMEEvaluator):
    """AIME 2026 benchmark."""

    benchmark_name = "aime-2026"
    dataset_path = "MathArena/aime_2026"
