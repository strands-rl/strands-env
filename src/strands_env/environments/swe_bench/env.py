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

"""SWE-bench environment using Harbor for container management and test execution.

A SWE-bench task is structurally identical to a Terminal-Bench task: a
Harbor task directory with `task.toml`, `instruction.md`,
`environment/Dockerfile`, `tests/test.sh`. The agent gets a single
`execute_command` tool and is expected to fix the repository at
`/testbed`. The reward script (`tests/test.sh`) runs the SWE-bench
test suite and writes `reward.txt`.

We thinly subclass `TerminalBenchEnv` to swap in a SWE-bench-tuned
system prompt; everything else (Docker/EKS backend selection, the
`execute_command` tool, the test-execution reward) is shared.
"""

from __future__ import annotations

from pathlib import Path

from typing_extensions import Unpack

from strands_env.core import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.environments.terminal_bench import TerminalBenchConfig, TerminalBenchEnv

#: SWE-bench reuses the Terminal-Bench config schema unchanged.
SWEBenchConfig = TerminalBenchConfig


class SWEBenchEnv(TerminalBenchEnv):
    """SWE-bench environment — Terminal-Bench env with a SWE-bench system prompt."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[SWEBenchConfig],
    ):
        """Initialize a `SWEBenchEnv` instance.

        Reuses `TerminalBenchReward` for verification — both benchmarks
        upload `tests/` to `/tests` and execute `test.sh`.
        """
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)
