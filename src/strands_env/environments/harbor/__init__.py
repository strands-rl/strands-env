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

"""Harbor task environment for Docker/e2b-based task evaluation."""

from pathlib import Path

from .env import HarborConfig, HarborEnv
from .reward import HarborReward

#: SWE-bench-tuned system prompt shipped with the env; benchmarks inject it via the
#: serializable `system_prompt` config key (e.g. the `swebench-verified` evaluator).
SWE_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt_swe.md"

__all__ = ["SWE_SYSTEM_PROMPT_PATH", "HarborConfig", "HarborEnv", "HarborReward"]
