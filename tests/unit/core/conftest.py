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

"""Fixtures scoped to core/ tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from strands_sglang import TokenManager

from strands_env.core.environment import Environment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_event_loop_metrics(
    cycle_count: int = 1,
    input_tokens: int = 10,
    output_tokens: int = 5,
    durations: list[float] | None = None,
) -> MagicMock:
    """Create a mock EventLoopMetrics with a single invocation cycle."""
    cycle = MagicMock()
    cycle.usage = {"inputTokens": input_tokens, "outputTokens": output_tokens}
    invocation = MagicMock()
    invocation.cycles = [cycle]

    metrics = MagicMock()
    metrics.cycle_count = cycle_count
    metrics.agent_invocations = [invocation]
    metrics.cycle_durations = durations if durations is not None else [0.1]
    metrics.tool_metrics = {}
    return metrics


def mock_agent(messages: list | None = None, event_loop_metrics: MagicMock | None = None) -> MagicMock:
    """Create a mock Agent instance with standard async methods."""
    agent_instance = MagicMock()
    agent_instance.invoke_async = AsyncMock()
    agent_instance.messages = messages if messages is not None else []
    agent_instance.model.token_manager = TokenManager()
    agent_instance.model.routed_experts = None
    agent_instance.event_loop_metrics = event_loop_metrics or mock_event_loop_metrics()
    return agent_instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.token_manager = TokenManager()
    return model


@pytest.fixture
def model_factory(mock_model):
    return lambda: mock_model


@pytest.fixture
def env(model_factory):
    return Environment(model_factory=model_factory)
