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

"""Simple math environment using a calculator tool."""

from pathlib import Path
from typing import Unpack, override

from strands_tools.calculator import calculator

from strands_env.core import Environment, EnvironmentConfig, ModelFactory, RewardFunction
from strands_env.environments.calculator.reward import MathVerifyReward


class CalculatorEnv(Environment):
    """Simple math environment using a calculator tool."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[EnvironmentConfig],
    ):
        """Initialize a `CalculatorEnv` instance."""
        super().__init__(model_factory=model_factory, reward_fn=reward_fn or MathVerifyReward(), **config)

    @override
    def get_tools(self) -> list:
        return [calculator]
