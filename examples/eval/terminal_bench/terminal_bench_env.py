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

"""Environment hook for terminal-bench-2 evaluation with `TerminalBenchEnv`."""

from __future__ import annotations

from strands_env.core.models import build_model_factory
from strands_env.core.types import Action
from strands_env.environments.terminal_bench import TerminalBenchEnv


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `TerminalBenchEnv`."""
    model_factory = build_model_factory(model_config)

    async def env_factory(action: Action) -> TerminalBenchEnv:
        """Create a new TerminalBenchEnv with its own Docker container."""
        ctx = action.task_context
        return TerminalBenchEnv(model_factory=model_factory, **ctx.config, **env_config)

    return env_factory
