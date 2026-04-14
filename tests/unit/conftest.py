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

"""Shared fixtures and helpers for unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from strands_env.core import Action, Environment, Observation, RewardResult, StepResult, TaskContext
from strands_env.eval import EvalSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample(reward: float, idx: int = 0, aborted: bool = False) -> EvalSample:
    """Create an EvalSample with the given reward and optional abort flag."""
    return EvalSample(
        action=Action(message="q", task_context=TaskContext(id=f"sample_{idx}")),
        step_result=StepResult(observation=Observation(), reward=RewardResult(reward=reward)),
        aborted=aborted,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env():
    """Mock Environment with async reset/step/cleanup."""
    env = MagicMock(spec=Environment)
    env.reset = AsyncMock()
    env.step = AsyncMock()
    env.cleanup = AsyncMock()
    return env


@pytest.fixture
def runner():
    """Click CliRunner for CLI tests."""
    return CliRunner()
