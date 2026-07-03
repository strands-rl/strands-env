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

"""Environment hook for tau2-bench evaluation with `Tau2BenchEnv`.

The agent under test comes from the CLI model flags. The user-simulator and the
NL-assertion judge default to `DEFAULT_USER_JUDGE_MODEL_CONFIG`; override them (or
disable the judge with `null`) and set `max_steps` via ``--env-config``:

    {
        "user_model_config":  { ...ModelConfig... },   # optional: user-simulator
        "judge_model_config": { ...ModelConfig... },   # optional: NL-assertion judge (null to disable)
        "max_steps": 100                               # optional, default 100 (tau2 step semantics)
    }
"""

from __future__ import annotations

from typing import Any

from strands_env.core.models import build_model_factory
from strands_env.core.types import Task
from strands_env.environments.tau2_bench import Tau2BenchEnv

#: Default model for the user-simulator and NL-assertion judge (override via ``--env-config``).
DEFAULT_USER_JUDGE_MODEL_CONFIG = {"backend": "bedrock", "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"}


def create_env_factory(model_config: dict, **env_config: Any):
    """Create an `env_factory` for `Tau2BenchEnv` (agent model from `model_config`)."""
    agent_model_factory = build_model_factory(model_config)
    user_model_factory = build_model_factory(env_config.pop("user_model_config", DEFAULT_USER_JUDGE_MODEL_CONFIG))
    judge_config = env_config.pop("judge_model_config", DEFAULT_USER_JUDGE_MODEL_CONFIG)
    judge_model_factory = build_model_factory(judge_config) if judge_config else None

    async def env_factory(task: Task) -> Tau2BenchEnv:
        """Construct a fresh `Tau2BenchEnv` for one task."""
        config = dict(task.config)  # type: ignore[attr-defined]
        return Tau2BenchEnv(
            agent_model_factory=agent_model_factory,
            user_model_factory=user_model_factory,
            judge_model_factory=judge_model_factory,
            **config,
            **env_config,
        )

    return env_factory
