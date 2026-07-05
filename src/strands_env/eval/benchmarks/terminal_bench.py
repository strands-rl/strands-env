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

"""Evaluator for Terminal-Bench (Harbor) benchmarks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import override

from harbor.models.task.task import Task as HarborTask

from strands_env.core import Task
from strands_env.environments.harbor import HarborConfig
from strands_env.eval import Evaluator
from strands_env.eval.evaluator import EvalSample

from ..registry import register_eval


class TerminalBenchTask(Task):
    """`Task` carrying the Harbor task config."""

    config: HarborConfig


class TerminalBenchEvaluator(Evaluator):
    """Base evaluator for Harbor-format benchmarks (loads a directory of task subdirs)."""

    tasks_subdir: str = "."  # subdirectory within `data_dir` if any
    system_prompt_path: Path | None = None  # optional benchmark-specific system prompt

    @property
    def data_dir(self) -> Path:
        """Local directory the dataset is downloaded into (`./data/<benchmark_name>`)."""
        return Path("data") / self.benchmark_name

    def _download_dataset(self) -> None:
        """Download Terminal-Bench tasks from Git repository."""
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.git_url, str(self.data_dir)],
            check=True,
        )

    @override
    def load_dataset(self) -> list[Task]:
        """Load Harbor-format tasks."""
        if not self.data_dir.exists():
            self._download_dataset()

        tasks = []
        for task_dir in sorted((self.data_dir / self.tasks_subdir).iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                tasks.append(self._load_single_task(task_dir))
        return tasks

    def _load_single_task(self, task_dir: Path) -> Task:
        """Load a single task from a directory."""
        task = HarborTask(task_dir)
        task.config.environment.memory_mb *= 2
        config: HarborConfig = {
            "task_id": task.name,
            "task_dir": str(task_dir.resolve()),
            "trial_dir": str(self.output_path.parent / task.name),
            "task_env_config": task.config.environment,
            "timeout": int(task.config.verifier.timeout_sec),
        }
        if self.system_prompt_path is not None:
            config["system_prompt"] = self.system_prompt_path.read_text().strip()
        return TerminalBenchTask(
            id=task.name,
            message=task.instruction,
            config=config,
        )

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where reward is missing or verification failed, so they are retried on resume."""
        reward_result = sample.result.reward_result
        if reward_result is None:
            # no reward_fn configured — deterministic, retrying can't help (run() warns about it)
            return True
        return reward_result.info.get("status") != "error"

    @override
    async def evaluate_sample(self, task: Task) -> EvalSample:
        """Override to create sample-specific output directories for pass@k."""
        assert isinstance(task, TerminalBenchTask)
        sample_idx = int(task.id.rsplit("_", 1)[1]) if "_" in task.id else 0
        task.config["trial_dir"] = str(self.output_path.parent / task.config["task_id"] / str(sample_idx))

        sample = await super().evaluate_sample(task)

        # Save agent messages
        agent_dir = Path(task.config["trial_dir"]) / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "messages.json").write_text(json.dumps(sample.result.messages, indent=2, default=str))
        return sample


@register_eval("terminal-bench-2.1")
class TerminalBench21Evaluator(TerminalBenchEvaluator):
    """Evaluator for Terminal-Bench-2.1 benchmark.

    The upstream repo nests its 89 Harbor-format tasks under a `tasks/` directory (alongside a
    `configs/` folder of leaderboard agent configs), so the task root differs from Terminal-Bench-2.
    """

    git_url = "https://github.com/harbor-framework/terminal-bench-2-1.git"
    tasks_subdir = "tasks"


@register_eval("terminal-bench-2")
class TerminalBench2Evaluator(TerminalBenchEvaluator):
    """Evaluator for Terminal-Bench-2 benchmark."""

    git_url = "https://github.com/laude-institute/terminal-bench-2.git"


@register_eval("terminal-bench-1")
class TerminalBench1Evaluator(TerminalBenchEvaluator):
    """Evaluator for Terminal-Bench-1 benchmark (migrated to Harbor format)."""

    git_url = "https://github.com/laude-institute/terminal-bench.git"

    def _rename_solution_yaml_files(self, tasks_dir: Path) -> None:
        """Rename solution.yaml files to allow all tasks to be mapped successfully."""
        for solution_yaml in tasks_dir.glob("*/solution.yaml"):
            solution_yaml.rename(solution_yaml.with_suffix(".yaml.bak"))

    @override
    def load_dataset(self) -> list[Task]:
        """Load and migrate Terminal Bench-1 tasks to Harbor format."""
        if not self.data_dir.exists():
            self._download_dataset()

        # Migrate to .harbor subdirectory
        migrated_dir = self.data_dir / ".harbor"
        if not migrated_dir.exists() or not any(migrated_dir.iterdir()):
            from harbor.mappers.terminal_bench import TerminalBenchMapper

            tasks_dir = self.data_dir / "original-tasks"
            self._rename_solution_yaml_files(tasks_dir)
            TerminalBenchMapper().map(tasks_dir, migrated_dir)

        tasks = []
        for task_dir in sorted(migrated_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                tasks.append(self._load_single_task(task_dir))
        return tasks
