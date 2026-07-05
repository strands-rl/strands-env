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

"""Environment hook for SWE-bench Verified evaluation with `HarborEnv`.

The `swebench-verified` benchmark injects a SWE-bench-tuned `system_prompt`
into each task's config, so this hook just instantiates the generic `HarborEnv`.
"""

from __future__ import annotations

from strands_env.core.models import build_model_factory
from strands_env.core.types import Task
from strands_env.environments.harbor import HarborEnv


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `HarborEnv`."""
    model_factory = build_model_factory(model_config)

    async def env_factory(_task: Task) -> HarborEnv:
        """Create a new HarborEnv with its own container/pod. The sample itself arrives via `rollout(task)`."""
        return HarborEnv(model_factory=model_factory, **env_config)

    return env_factory
