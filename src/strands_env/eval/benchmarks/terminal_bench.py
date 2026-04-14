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

from harbor.models.task.task import Task
from typing_extensions import override

from strands_env.core import Action, TaskContext
from strands_env.environments.terminal_bench import TerminalBenchConfig
from strands_env.eval import Evaluator
from strands_env.eval.evaluator import EvalSample

from ..registry import register_eval


class TerminalBenchTaskContext(TaskContext):
    """TaskContext with Terminal-Bench specific fields."""

    config: TerminalBenchConfig


class TerminalBenchEvaluator(Evaluator):
    """Base evaluator for Terminal-Bench benchmarks."""

    GIT_URL: str = ""
    data_dir: Path = Path("./data/terminal-bench")

    def _download_dataset(self) -> None:
        """Download Terminal-Bench tasks from Git repository."""
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", self.GIT_URL, str(self.data_dir)],
            check=True,
        )

    @override
    def load_dataset(self) -> list[Action]:
        """Load Harbor-format tasks and create Actions."""
        if not self.data_dir.exists():
            self._download_dataset()

        actions = []
        for task_dir in sorted(self.data_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                actions.append(self._load_single_task(task_dir))
        return actions

    def _load_single_task(self, task_dir: Path) -> Action:
        """Load a single task from a directory."""
        task = Task(task_dir)
        config: TerminalBenchConfig = {
            "task_id": task.name,
            "task_dir": str(task_dir.resolve()),
            "trial_dir": str(self.output_path.parent / task.name),
            "harbor_env_config": task.config.environment,
            "timeout": int(task.config.verifier.timeout_sec),
        }
        return Action(
            message=task.instruction,
            task_context=TerminalBenchTaskContext(id=task.name, config=config),
        )

    @override
    async def evaluate_sample(self, action: Action) -> EvalSample:
        """Override to create sample-specific output directories for pass@k."""
        assert isinstance(action.task_context, TerminalBenchTaskContext)
        ctx = action.task_context
        sample_idx = int(ctx.id.rsplit("_", 1)[1]) if "_" in ctx.id else 0
        ctx.config["trial_dir"] = str(self.output_path.parent / ctx.config["task_id"] / str(sample_idx))

        sample = await super().evaluate_sample(action)

        # Save agent messages
        agent_dir = Path(ctx.config["trial_dir"]) / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "messages.json").write_text(
            json.dumps(sample.step_result.observation.messages, indent=2, default=str)
        )
        return sample


@register_eval("terminal-bench-2")
class TerminalBench2Evaluator(TerminalBenchEvaluator):
    """Evaluator for Terminal-Bench-2 benchmark."""

    benchmark_name = "terminal-bench-2"
    GIT_URL = "https://github.com/laude-institute/terminal-bench-2.git"
    data_dir: Path = Path("./data/terminal-bench-2")


@register_eval("terminal-bench-1")
class TerminalBench1Evaluator(TerminalBenchEvaluator):
    """Evaluator for Terminal-Bench-1 benchmark (migrated to Harbor format)."""

    benchmark_name = "terminal-bench-1"
    GIT_URL = "https://github.com/laude-institute/terminal-bench.git"
    data_dir: Path = Path("./data/terminal-bench-1")

    def _rename_solution_yaml_files(self, tasks_dir: Path) -> None:
        """Rename solution.yaml files to allow all tasks to be mapped successfully."""
        for solution_yaml in tasks_dir.glob("*/solution.yaml"):
            solution_yaml.rename(solution_yaml.with_suffix(".yaml.bak"))

    @override
    def load_dataset(self) -> list[Action]:
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

        actions = []
        for task_dir in sorted(migrated_dir.iterdir()):
            if task_dir.is_dir() and not task_dir.name.startswith("."):
                actions.append(self._load_single_task(task_dir))
        return actions
