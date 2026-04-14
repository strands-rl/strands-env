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

"""Evaluation metrics for benchmark results."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import EvalSample

#: Type alias for metric function: takes results {prompt_id: [EvalSample, ...]}, returns {metric_name: value}.
MetricFn = Callable[[dict[str, list["EvalSample"]]], dict[str, float]]


def compute_pass_at_k(
    results: dict[str, list[EvalSample]],
    k_values: list[int],
    reward_threshold: float = 1.0,
) -> dict[str, float]:
    """Compute pass@k metrics using unbiased estimator.

    Args:
        results: Dict mapping prompt_id to list of samples.
        k_values: List of k values for pass@k.
        reward_threshold: Reward threshold for "pass" (default: 1.0).

    Returns:
        Dict mapping "pass@k" to average score.
    """
    if not results:
        return {f"pass@{k}": 0.0 for k in k_values}

    def is_correct(s: EvalSample) -> bool:
        r = s.step_result.reward
        return r is not None and r.reward >= reward_threshold

    def pass_at_k_single(n: int, c: int, k: int) -> float:
        """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
        if n - c < k:
            return 1.0
        if c == 0:
            return 0.0
        log_ratio = sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
        return 1.0 - math.exp(log_ratio)

    metrics = {}
    for k in k_values:
        scores = []
        for samples in results.values():
            n, c = len(samples), sum(1 for s in samples if is_correct(s))
            if k <= n:
                scores.append(pass_at_k_single(n, c, k))
        metrics[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0

    return metrics
