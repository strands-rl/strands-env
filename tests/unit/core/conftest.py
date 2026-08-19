"""Fixtures scoped to core/ tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from strands_sglang import Rollout

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
    agent_instance.model.rollout = Rollout()
    agent_instance.event_loop_metrics = event_loop_metrics or mock_event_loop_metrics()
    return agent_instance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.rollout = Rollout()
    return model


@pytest.fixture
def model_factory(mock_model):
    return lambda: mock_model


@pytest.fixture
def env(model_factory):
    return Environment(model_factory=model_factory)
