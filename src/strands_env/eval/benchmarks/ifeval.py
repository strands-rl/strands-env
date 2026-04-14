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

"""Evaluator for IFEval (Instruction-Following Eval) benchmark."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from datasets import load_dataset
from lm_eval.tasks.ifeval.utils import process_results
from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.core.types import RewardFunction, RewardResult, StepResult

from ..evaluator import Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)

#: Canonical IFEval metrics produced by `lm_eval.tasks.ifeval.utils.process_results`.
IFEVAL_METRICS = (
    "prompt_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_strict_acc",
    "inst_level_loose_acc",
)


class IFEvalReward(RewardFunction):
    """Rule-based reward for IFEval.

    Runs the response through every verifier referenced by the sample's
    `instruction_id_list` and returns the canonical four IFEval metrics.
    The scalar reward defaults to `prompt_level_strict_acc` (1.0 only if
    **all** instructions are followed), but can be swapped via `metric`.
    All four metrics and the per-instruction boolean lists are attached
    to `info`.
    """

    def __init__(self, metric: str = "prompt_level_strict_acc") -> None:
        """Initialize an `IFEvalReward` instance.

        Args:
            metric: Which IFEval metric to use as the scalar reward. Must be
                one of `IFEVAL_METRICS`. `inst_level_*` metrics are collapsed
                to a scalar via the mean of the per-instruction booleans.
        """
        if metric not in IFEVAL_METRICS:
            raise ValueError(f"Unknown IFEval metric {metric!r}; expected one of {IFEVAL_METRICS}")
        self.metric = metric

    @staticmethod
    def _scalar(value: bool | list[bool]) -> float:
        """Collapse a strict/loose metric value to a float in [0, 1]."""
        if isinstance(value, bool):
            return float(value)
        return (sum(value) / len(value)) if value else 0.0

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run the IFEval grader on the final response and return a `RewardResult`."""
        response = step_result.observation.final_response or ""
        ctx = action.task_context

        # `process_results` expects a `doc` dict mirroring the original IFEval JSONL row.
        doc: dict[str, Any] = {
            "key": getattr(ctx, "key", 0),
            "instruction_id_list": list(getattr(ctx, "instruction_id_list", [])),
            "prompt": action.message if isinstance(action.message, str) else "",
            "kwargs": list(getattr(ctx, "ifeval_kwargs", [])),
        }

        try:
            results = process_results(doc, [response])
        except Exception as e:
            logger.error("IFEval grader failed for %s: %s", ctx.id, e)
            return RewardResult(reward=0.0, info={"status": "error", "error": str(e)})

        info: dict[str, Any] = {
            "status": "success",
            "prompt_level_strict": float(results["prompt_level_strict_acc"]),
            "prompt_level_loose": float(results["prompt_level_loose_acc"]),
            "inst_level_strict": self._scalar(results["inst_level_strict_acc"]),
            "inst_level_loose": self._scalar(results["inst_level_loose_acc"]),
            "per_instruction_strict": list(results["inst_level_strict_acc"]),
            "per_instruction_loose": list(results["inst_level_loose_acc"]),
        }
        return RewardResult(reward=self._scalar(results[self.metric]), info=info)


@register_eval("ifeval")
class IFEvalEvaluator(Evaluator):
    """Evaluator for IFEval (Instruction-Following Eval).

    Loads the 541-row [google/IFEval](https://huggingface.co/datasets/google/IFEval)
    dataset and emits one `Action` per row. Each `TaskContext` carries the `key`,
    `instruction_id_list`, and `ifeval_kwargs` fields needed by `IFEvalReward` to
    run the rule-based grader. Pair this evaluator with an `IFEvalReward` instance
    on the environment.
    """

    benchmark_name: str = "ifeval"
    dataset_path: str = "google/IFEval"

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load the IFEval dataset from HuggingFace (streaming).

        Yields:
            Action objects. Each `TaskContext` stores the IFEval `key`,
            `instruction_id_list`, and `ifeval_kwargs` as extras.
        """
        dataset = load_dataset(self.dataset_path, split="train", streaming=True)

        for i, row in enumerate(dataset):
            prompt = row.get("prompt")
            if not prompt:
                logger.warning("Row %s: missing prompt, skipped", i)
                continue
            extras: dict[str, Any] = {
                "key": int(row.get("key", i)),
                "instruction_id_list": list(row.get("instruction_id_list", [])),
                "ifeval_kwargs": list(row.get("kwargs", [])),
            }
            yield Action(
                message=str(prompt),
                task_context=TaskContext(id=f"{self.benchmark_name}_{row.get('key', i)}", **extras),
            )
