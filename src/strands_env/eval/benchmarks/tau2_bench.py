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

import importlib
import logging
import math
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.environments.tau2_bench import Tau2BenchConfig
from strands_env.eval import Evaluator
from strands_env.eval.evaluator import EvalSample
from strands_env.eval.metrics import MetricFn, compute_pass_at_k

from ..registry import register_eval

# tau2 freezes its data dir from TAU2_DATA_DIR at import; must set it before importing tau2.
DATA_DIR = Path("./data/tau2-bench")
os.environ.setdefault("TAU2_DATA_DIR", str((DATA_DIR / "data").resolve()))

from tau2.user.user_simulator import get_global_user_sim_guidelines  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


def compute_pass_caret_k(
    results: dict[str, list[EvalSample]],
    k_values: list[int],
    reward_threshold: float = 1.0,
) -> dict[str, float]:
    """Consistency metric ``pass^k = C(c, k) / C(n, k)`` averaged across prompts."""

    def is_correct(s: EvalSample) -> bool:
        r = s.step_result.reward
        return r is not None and r.reward >= reward_threshold

    metrics = {}
    for k in k_values:
        scores = []
        for samples in results.values():
            n = len(samples)
            c = sum(1 for s in samples if is_correct(s))
            if k > n:  # keep: math.comb(n, k) would be 0 here and divide by zero
                continue
            scores.append(math.comb(c, k) / math.comb(n, k))
        metrics[f"pass^{k}"] = sum(scores) / len(scores) if scores else 0.0
    return metrics


class Tau2BenchTaskContext(TaskContext):
    """`TaskContext` for tau2-bench."""

    config: Tau2BenchConfig


class Tau2BenchEvaluator(Evaluator):
    """Base evaluator for tau2-bench; subclasses set `domain` and `user_has_tools`."""

    git_url: str = "https://github.com/sierra-research/tau2-bench.git"
    # Tag to clone the data files from.
    git_ref: str = "v1.0.0"
    data_dir: Path = DATA_DIR

    domain: ClassVar[Literal["airline", "retail", "telecom"]]
    user_has_tools: ClassVar[bool] = False

    def _download_dataset(self) -> None:
        """Clone tau2-bench data files at `git_ref` (not bundled with the `tau2` pip wheel)."""
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", self.git_ref, self.git_url, str(self.data_dir)],
            check=True,
        )

    @override
    def load_dataset(self) -> list[Action]:
        """Enumerate tasks for `self.domain` and bundle statics into each Action."""
        if not self.data_dir.exists():
            self._download_dataset()
        domain_mod = importlib.import_module(f"tau2.domains.{self.domain}.environment")
        tasks = [task.model_dump(mode="json") for task in domain_mod.get_tasks(task_split_name="base")]
        user_sim_guidelines = get_global_user_sim_guidelines(use_tools=self.user_has_tools)
        return [
            Action(
                message="",  # set by Tau2BenchEnv.step() from env.first_user_msg (after reset)
                task_context=Tau2BenchTaskContext(
                    id=str(task["id"]),
                    ground_truth=(task.get("evaluation_criteria") or {}).get("reward_basis"),
                    config=Tau2BenchConfig(
                        domain=self.domain,
                        task=task,
                        user_sim_guidelines=user_sim_guidelines,
                    ),
                ),
            )
            for task in tasks
        ]

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples with missing reward, NL judge error, or `tau2_termination == "aborted"` (all retryable)."""
        reward = sample.step_result.reward
        if reward is None:
            return False
        nl_judge = reward.info.get("nl_judge")
        if nl_judge and nl_judge.get("status") == "error":
            return False
        if (sample.step_result.observation.metrics or {}).get("tau2_termination") == "aborted":
            return False
        return True

    @override
    def get_metric_fns(self) -> list[MetricFn]:
        """Report both ``pass@k`` (at-least-one) and ``pass^k`` (consistency, tau2 paper)."""
        k_values = list(range(1, self.n_samples_per_prompt + 1))
        return [
            partial(compute_pass_at_k, k_values=k_values, reward_threshold=1.0),
            partial(compute_pass_caret_k, k_values=k_values, reward_threshold=1.0),
        ]


@register_eval("tau2-bench-retail")
class Tau2BenchRetailEvaluator(Tau2BenchEvaluator):
    """tau2-bench retail domain (114 tasks)."""

    benchmark_name = "tau2-bench-retail"
    domain: ClassVar[Literal["airline", "retail", "telecom"]] = "retail"


@register_eval("tau2-bench-airline")
class Tau2BenchAirlineEvaluator(Tau2BenchEvaluator):
    """tau2-bench airline domain (50 tasks)."""

    benchmark_name = "tau2-bench-airline"
    domain: ClassVar[Literal["airline", "retail", "telecom"]] = "airline"


@register_eval("tau2-bench-telecom")
class Tau2BenchTelecomEvaluator(Tau2BenchEvaluator):
    """tau2-bench telecom domain (114 tasks, sub-sampled from 2285)."""

    benchmark_name = "tau2-bench-telecom"
    domain: ClassVar[Literal["airline", "retail", "telecom"]] = "telecom"
    user_has_tools: ClassVar[bool] = True
