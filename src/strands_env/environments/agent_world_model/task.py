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

"""The per-sample input type for `AgentWorldModelEnv`."""

from __future__ import annotations

from pydantic import Field

from strands_env.core import Task


class AgentWorldModelTask(Task):
    """Rollout input for `AgentWorldModelEnv`."""

    scenario: str = Field(description="Scenario name in `gen_envs.jsonl` (which synthetic world to boot).")
    task_idx: int = Field(description="Task index within the scenario (used in logs).")
    verify_code: str = Field(description="Python source defining `verify_task_completion(...)`.")
    initial_db_path: str = Field(
        description="Pristine SQLite snapshot for this scenario; the episode's working DB is cloned from it."
    )
