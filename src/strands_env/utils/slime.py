# Copyright 2025-2026 Horizon RL Contributors
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

"""Utilities for logging `strands-env` metrics in `slime`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import wandb  # type: ignore

if TYPE_CHECKING:
    from slime.utils.types import Sample  # type: ignore

logger = logging.getLogger(__name__)

# Make sure nested metrics are logged with the correct default "rollout/step".
wandb.define_metric("rollout/*/*", step_metric="rollout/step")


def collect_env_metrics(samples: list[Sample]) -> dict[str, float]:
    """Aggregate strands-env observation metrics across samples.

    Extracts metrics from `sample.metrics` (populated by `Environment.step()`)
    and returns aggregated statistics (mean/median/max/min) suitable for logging.

    Args:
        samples: List of slime `Sample` objects with a `metrics` dict attribute.

    Returns:
        Dict with `{name}/mean`, `{name}/median`, `{name}/max`, `{name}/min`
        for each metric, plus `tool_success_rate` and `tool_parse_error_rate`.
    """
    from slime.utils.metric_utils import compute_statistics, dict_add_prefix  # type: ignore

    if not samples:
        return {}

    per_sample: dict[str, list[float]] = {
        "message_count": [],
        "tool_iters": [],
        "tool_calls": [],
        "model_calls": [],
        "model_latency_s": [],
        "tool_latency_s": [],
        "cache_hit_rate": [],
    }
    total_tool_successes = 0
    total_tool_calls = 0
    total_parse_errors = 0

    for sample in samples:
        # NOTE: need to set sample.metrics = step_result.observation.metrics in generate()
        metrics: dict[str, Any] = getattr(sample, "metrics", None) or {}

        per_sample["message_count"].append(metrics.get("message_count", 0))
        per_sample["tool_iters"].append(metrics.get("tool_iters", 0))
        per_sample["tool_calls"].append(metrics.get("tool_calls", 0))
        per_sample["model_calls"].append(metrics.get("model_calls", 0))
        latency = metrics.get("model_latency_s")
        per_sample["model_latency_s"].append(latency["total"] if latency else 0)
        per_sample["cache_hit_rate"].append(metrics.get("cache_hit_rate") or 0)

        per_tool = metrics.get("per_tool_metrics") or {}
        per_sample["tool_latency_s"].append(sum(tm["latency_s"] for tm in per_tool.values()))
        total_tool_calls += sum(tm["calls"] for tm in per_tool.values())
        total_tool_successes += sum(tm["successes"] for tm in per_tool.values())
        total_parse_errors += sum(tm.get("parse_errors", 0) for tm in per_tool.values())

    log_dict: dict[str, float] = {}
    for name, values in per_sample.items():
        log_dict |= dict_add_prefix(compute_statistics(values), f"{name}/")

    if total_tool_calls > 0:
        log_dict["tool_success_rate"] = total_tool_successes / total_tool_calls
        log_dict["tool_parse_error_rate"] = total_parse_errors / total_tool_calls

    return log_dict


def log_rollout_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    _rollout_extra_metrics: dict | None,
    _rollout_time: float,
) -> bool:
    """Custom rollout log function for slime.

    Extracts strands-env environment metrics from samples and logs them alongside
    slime's default metrics. Returns `False` so slime's default logging still runs.

    Args:
        rollout_id: Current rollout iteration number.
        args: slime training arguments.
        samples: List of samples from the rollout.
        _rollout_extra_metrics: Additional metrics from rollout pipeline (unused).
        _rollout_time: Wall-clock time for the rollout (unused).

    Returns:
        `False` to continue with default slime logging.
    """
    from slime.utils import logging_utils  # type: ignore
    from slime.utils.metric_utils import compute_rollout_step, dict_add_prefix  # type: ignore

    log_dict = collect_env_metrics(samples)
    if not log_dict:
        return False

    step = compute_rollout_step(args, rollout_id)
    log_dict = dict_add_prefix(log_dict, "rollout/")
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")

    logger.debug("Logged strands-env metrics for rollout %s: %s", rollout_id, list(log_dict.keys()))

    return False
