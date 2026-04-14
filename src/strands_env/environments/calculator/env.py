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

from strands_tools.calculator import calculator
from typing_extensions import override

from strands_env.core.environment import Environment


class CalculatorEnv(Environment):
    """Simple math environment using a calculator tool."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    @override
    def get_tools(self) -> list:
        return [calculator]
