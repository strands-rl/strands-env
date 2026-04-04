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

"""Example environment hook for math reasoning evaluation with `CalculatorEnv`."""

from strands_env.core.models import build_model_factory
from strands_env.environments.calculator import CalculatorEnv
from strands_env.rewards import MathVerifyReward


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `CalculatorEnv`."""
    model_factory = build_model_factory(model_config)
    reward_fn = MathVerifyReward()

    async def env_factory(_action):
        return CalculatorEnv(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
