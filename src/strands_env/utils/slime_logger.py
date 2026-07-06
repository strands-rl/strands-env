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

"""Rollout logger for `slime` training, with a pluggable logging backend.

`RolloutLogger` aggregates per-rollout environment metrics into slime's
`rollout_extra_metrics` and publishes a sample of decoded rollouts to the
configured `backend`:

- `"wandb"` — metrics to `wandb`, samples to a W&B Weave dataset.
- `"mlflow"` — metrics and samples to MLflow.

Instantiate one and pass its bound `log_rollouts` as slime's
`--custom-rollout-log-function-path` callback. Backend libraries are imported
lazily, so only the selected backend needs to be installed.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, Literal

from slime.rollout.sglang_rollout import GenerateState  # type: ignore
from slime.utils.metric_utils import compute_rollout_step, compute_statistics, dict_add_prefix  # type: ignore
from slime.utils.types import Sample  # type: ignore

from strands_env.core.types import RolloutResult

if TYPE_CHECKING:
    from weave.trace.refs import ObjectRef
    from weave.trace.weave_client import WeaveClient

logger = logging.getLogger(__name__)

# NOTE: response_len in default logging refers to loss_mask=1 tokens


class RolloutLogger:
    """Custom `slime` rollout logger with a pluggable logging backend.

    Aggregates per-rollout env metrics into slime's `rollout_extra_metrics` and
    publishes a sample of decoded rollouts to `backend` (`"wandb"` → a W&B Weave
    dataset, `"mlflow"` → MLflow JSON artifacts).

    Instantiate once and use `log_rollouts` as the
    `--custom-rollout-log-function-path` callback. Sample logging is controlled by
    `n_rollouts_per_step` (default 3); set to 0 to disable. `max_rollouts` caps the
    accumulated Weave dataset and is ignored by the MLflow backend.
    """

    def __init__(
        self,
        backend: Literal["wandb", "mlflow"] = "wandb",
        n_rollouts_per_step: int = 3,
        max_rollouts: int = 3000,
        log_per_tool_metrics: bool = False,
    ) -> None:
        self.backend = backend
        self.n_rollouts_per_step = n_rollouts_per_step
        self.log_per_tool_metrics = log_per_tool_metrics
        self.max_rollouts = max_rollouts
        # Weave (wandb backend) state, lazily initialized on first publish.
        self._weave_init = False
        self._weave_client: WeaveClient | None = None
        self._prev_ref: ObjectRef | None = None
        self._rows: list[dict] = []
        self.run_name: str | None = None

    def log_rollouts(
        self,
        rollout_id: int,
        args: Any,
        samples: list[Sample],
        rollout_extra_metrics: dict | None,
        _rollout_time: float,
    ) -> bool:
        """Log env metrics and optionally publish sampled rollouts.

        Returns `False` so slime's default logging still runs.
        """
        # Check if rollout results are attached to samples
        for sample in samples:
            if not getattr(sample, "result", None):
                logger.warning("Skip custom rollout logging for rollout %d: missing `result`", rollout_id)
                return False

        self.log_rollout_metrics(samples=samples, rollout_extra_metrics=rollout_extra_metrics)
        self.log_rollout_samples(rollout_id=rollout_id, args=args, samples=samples)

        return False

    def log_rollout_metrics(self, samples: list[Sample], rollout_extra_metrics: dict | None) -> None:
        """Aggregate `RolloutResult.metrics` across samples into `rollout_extra_metrics`.

        Note:
            - Need to set `sample.metrics = result.metrics` in `generate()`
            - Overrides for more custom rollout logging metrics can be added here
        """
        per_sample: dict[str, list[float]] = {
            "message_count": [],
            "model_calls": [],
            "model_latency_s": [],
            "cache_hit_rate": [],
            "tool_iters": [],
            "tool_calls": [],
            "executed_tool_calls": [],
            "cancelled_tool_calls": [],
            "tool_latency_s": [],
        }

        aggregated: dict[str, float] = {
            "tool_name_error_rate": 0.0,
            "tool_success_rate": 0.0,
            "tool_parse_error_rate": 0.0,
        }

        total_executed_tool_calls = 0
        for sample in samples:
            metrics: dict[str, Any] = sample.result.metrics
            if not metrics:
                continue

            per_sample["message_count"].append(metrics.get("message_count", 0))
            per_sample["tool_iters"].append(metrics.get("tool_iters", 0))
            per_sample["tool_calls"].append(metrics.get("tool_calls", 0))
            per_sample["cancelled_tool_calls"].append(metrics.get("cancelled_tool_calls", 0))
            per_sample["model_calls"].append(metrics.get("model_calls", 0))
            latency = metrics.get("model_latency_s")
            per_sample["model_latency_s"].append(latency["total"] if latency else 0)
            per_sample["cache_hit_rate"].append(metrics.get("cache_hit_rate") or 0)

            executed_tool_calls = 0
            tool_latency_s = 0.0
            for tool_name, tm in (metrics.get("per_tool_metrics") or {}).items():
                key = f"{tool_name}_tool"
                calls = tm["calls"]
                if tm["is_known"]:
                    if self.log_per_tool_metrics:
                        per_sample.setdefault(f"{key}_calls", []).append(calls)
                        per_sample.setdefault(f"{key}_latency_s", []).append(tm["latency_s"])
                        per_sample.setdefault(f"{key}_success_rate", []).append(tm["successes"] / calls)
                        per_sample.setdefault(f"{key}_parse_error_rate", []).append(tm.get("parse_errors", 0) / calls)
                else:
                    aggregated["tool_name_error_rate"] += calls
                executed_tool_calls += calls
                aggregated["tool_success_rate"] += tm["successes"]
                aggregated["tool_parse_error_rate"] += tm.get("parse_errors", 0)
                tool_latency_s += tm["latency_s"]
            total_executed_tool_calls += executed_tool_calls
            per_sample["executed_tool_calls"].append(executed_tool_calls)
            per_sample["tool_latency_s"].append(tool_latency_s / executed_tool_calls if executed_tool_calls else 0.0)

        log_dict: dict[str, float] = {}
        for name, values in per_sample.items():
            assert values, f"Empty values for per-sample metric: {name}"
            log_dict |= {f"{name}_{k}": v for k, v in compute_statistics(values).items()}
        for name, value in aggregated.items():
            log_dict[f"{name}"] = value / total_executed_tool_calls if total_executed_tool_calls else 0
        log_dict = dict_add_prefix(log_dict, "rollout/")
        if rollout_extra_metrics is not None:
            rollout_extra_metrics.update(log_dict)
        else:
            logger.warning("rollout_extra_metrics is None, env metrics will not be logged")

    def _build_sample_rows(self, rollout_id: int, step: int, args: Any, samples: list[Sample]) -> list[dict]:
        """Decode a random subset of rollouts into serializable rows for backend logging."""
        tokenizer = GenerateState(args).tokenizer
        n_saved = min(len(samples), self.n_rollouts_per_step)
        rows = []
        for s in random.sample(samples, k=n_saved):
            result: RolloutResult = s.result
            rollout = result.rollout
            if not rollout:
                logger.warning("rollout %d missing token rollout", rollout_id)
                continue

            prompt_len = rollout.initial_prompt_length
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "step": step,
                    "group_index": s.group_index,
                    "index": s.index,
                    "prompt": tokenizer.decode(rollout.token_ids[:prompt_len], skip_special_tokens=False),
                    "response": tokenizer.decode(rollout.token_ids[prompt_len:], skip_special_tokens=False),
                    "termination_reason": result.termination_reason.value,
                    "reward": result.reward_result.reward if result.reward_result else None,
                    "reward_info": result.reward_result.info if result.reward_result else None,
                    "metrics": result.metrics,
                }
            )
        return rows

    def log_rollout_samples(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish a sample of decoded rollouts to the configured backend."""
        match self.backend:
            case "wandb":
                self._log_samples_wandb(rollout_id, args, samples)
            case "mlflow":
                self._log_samples_mlflow(rollout_id, args, samples)
            case _:
                raise ValueError(f"Unknown logging backend {self.backend!r} (expected 'wandb' or 'mlflow')")

    def _log_samples_wandb(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish sampled rollout results to a single W&B Weave dataset per run."""
        import wandb
        import weave

        # Lazy Weave init from args.wandb_project
        if not self._weave_init:
            project = getattr(args, "wandb_project", None)
            if not project:
                return
            self._weave_client = weave.init(project)
            self.run_name = wandb.run.name if wandb.run else "unknown"
            self._weave_init = True

        step = compute_rollout_step(args, rollout_id)
        rows = self._build_sample_rows(rollout_id, step, args, samples)
        if not rows:
            return

        # Accumulate rows locally, cap at max_rollouts, publish fresh each time.
        self._rows.extend(rows)
        if len(self._rows) > self.max_rollouts:
            self._rows = self._rows[-self.max_rollouts :]

        dataset_name = f"{self.run_name}_rollouts"
        dataset = weave.Dataset(name=dataset_name, rows=weave.Table(rows=self._rows))
        new_ref = weave.publish(dataset)

        # Delete previous version (each version is a superset, so old ones are redundant).
        if self._prev_ref is not None and self._weave_client is not None:
            try:
                self._weave_client.delete_object_version(self._prev_ref)
            except Exception:
                logger.debug("Failed to delete previous Weave dataset version", exc_info=True)
        self._prev_ref = new_ref

        logger.info("Published %d new samples to Weave (rollout %d, step %d)", len(rows), rollout_id, step)

    def _log_samples_mlflow(self, rollout_id: int, args: Any, samples: list[Sample]) -> None:
        """Publish sampled rollout results to MLflow as a per-step JSON artifact."""
        import mlflow

        step = compute_rollout_step(args, rollout_id)
        rows = self._build_sample_rows(rollout_id, step, args, samples)
        if not rows:
            return

        mlflow.log_dict(rows, f"rollout_samples/step_{step:05d}.json")  # type: ignore[arg-type]

        logger.info("Logged %d samples to MLflow (rollout %d, step %d)", len(rows), rollout_id, step)
