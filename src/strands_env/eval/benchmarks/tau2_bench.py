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

"""Evaluator for tau2-bench (Sierra Research, multi-turn customer service)."""

from __future__ import annotations

import logging
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import Literal, override

from strands_env.core import Task
from strands_env.environments.tau2_bench import Tau2BenchTask, _tau2
from strands_env.eval import Evaluator
from strands_env.eval.evaluator import EvalSample
from strands_env.eval.metrics import MetricFunction, compute_pass_at_k, compute_pass_power_k

from ..registry import register_eval

# tau2 freezes its data dir from TAU2_DATA_DIR at import; must set it before importing tau2.
DATA_DIR = Path("./data/tau2-bench")
os.environ.setdefault("TAU2_DATA_DIR", str((DATA_DIR / "data").resolve()))

logger = logging.getLogger(__name__)


class Tau2BenchEvaluator(Evaluator):
    """Base evaluator for tau2-bench; subclasses set `domain`."""

    benchmark_name: str = "tau2-bench"
    domain: Literal["airline", "retail", "telecom"]
    git_url: str = "https://github.com/sierra-research/tau2-bench.git"
    git_ref: str = "v1.0.0"
    data_dir: Path = DATA_DIR

    def _download_dataset(self) -> None:
        """Clone tau2-bench data files at `git_ref` (not bundled with the `tau2` pip wheel)."""
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", self.git_ref, self.git_url, str(self.data_dir)],
            check=True,
        )

    @override
    def load_dataset(self) -> list[Task]:
        """Enumerate tasks for `self.domain` and bundle statics into each Task."""
        if not self.data_dir.exists():
            self._download_dataset()
        tasks = [task.model_dump(mode="json") for task in _tau2.get_tasks(self.domain)]
        return [
            Tau2BenchTask(
                id=str(task["id"]),
                domain=self.domain,
                config=task,
            )
            for task in tasks
        ]

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples with missing reward, NL judge error, or an aborted user-sim (all retryable)."""
        reward = sample.result.reward_result
        if reward is None:
            return False
        nl_judge = reward.info.get("nl_judge")
        if nl_judge and nl_judge.get("status") == "error":
            return False
        if ((sample.result.metrics or {}).get("user_simulator") or {}).get("termination") == "aborted":
            return False
        return True

    @override
    def get_metric_fns(self) -> list[MetricFunction]:
        """Report both ``pass@k`` (at-least-one) and ``pass^k`` (consistency, tau2 paper)."""
        k_values = list(range(1, self.n_samples_per_prompt + 1))
        return [
            partial(compute_pass_at_k, k_values=k_values, reward_threshold=1.0),
            partial(compute_pass_power_k, k_values=k_values, reward_threshold=1.0),
        ]


@register_eval("tau2-bench-retail")
class Tau2BenchRetailEvaluator(Tau2BenchEvaluator):
    """tau2-bench retail domain (114 tasks)."""

    benchmark_name = "tau2-bench-retail"
    domain = "retail"


@register_eval("tau2-bench-airline")
class Tau2BenchAirlineEvaluator(Tau2BenchEvaluator):
    """tau2-bench airline domain (50 tasks)."""

    benchmark_name = "tau2-bench-airline"
    domain = "airline"


@register_eval("tau2-bench-telecom")
class Tau2BenchTelecomEvaluator(Tau2BenchEvaluator):
    """tau2-bench telecom domain (114 tasks, sub-sampled from 2285)."""

    benchmark_name = "tau2-bench-telecom"
    domain = "telecom"
