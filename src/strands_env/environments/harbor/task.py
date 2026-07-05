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

"""The dataset-authored Harbor sample type."""

from __future__ import annotations

from pathlib import Path

from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task as HarborTaskSpec
from harbor.models.trial.paths import TrialPaths
from pydantic import Field

from strands_env.core import Task


class HarborTask(Task):
    """Rollout input for `HarborEnv`: one Harbor-format task bundle."""

    task_id: str = Field(description="Harbor task name (also keys e2b template lookups).")
    task_dir: str = Field(description="Path to the task bundle (task.toml, environment/, tests/).")
    trial_dir: str = Field(description="Output directory for this trial's artifacts (evaluator-authored layout).")
    verifier_timeout: int | None = Field(
        default=None, description="Verifier budget (seconds) from task.toml [verifier]; None = env's exec_timeout."
    )
    task_env_config: TaskEnvironmentConfig = Field(
        default_factory=TaskEnvironmentConfig, description="Harbor container settings from task.toml [environment]."
    )
    system_prompt: str | None = Field(default=None, description="Task-specific system prompt override (e.g. tb2, swe).")

    @property
    def task_paths(self) -> TaskPaths:
        """Harbor's view of the task bundle. A plain property — cheap, and never stale."""
        return TaskPaths(Path(self.task_dir))

    @property
    def trial_paths(self) -> TrialPaths:
        """Harbor's view of the trial output dir. Not cached: `trial_dir` is rewritten per pass@k sample."""
        return TrialPaths(Path(self.trial_dir))

    @classmethod
    def from_task_dir(
        cls, task_dir: str | Path, *, trial_dir: str | Path, system_prompt: str | None = None
    ) -> HarborTask:
        """Build a `HarborTask` from an on-disk Harbor task bundle (reads `task.toml`)."""
        spec = HarborTaskSpec(Path(task_dir))
        return cls(
            id=spec.name,
            message=spec.instruction,
            task_id=spec.name,
            task_dir=str(Path(task_dir).resolve()),
            trial_dir=str(trial_dir),
            task_env_config=spec.config.environment,
            verifier_timeout=int(spec.config.verifier.timeout_sec),
            system_prompt=system_prompt,
        )
