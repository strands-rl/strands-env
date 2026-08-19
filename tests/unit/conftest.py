from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from strands_env.core import Environment, RewardResult, RolloutResult, Task
from strands_env.eval import EvalSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sample(reward: float, idx: int = 0, aborted: bool = False) -> EvalSample:
    """Create an EvalSample with the given reward and optional abort flag."""
    return EvalSample(
        task=Task(id=f"sample_{idx}", message="q"),
        result=RolloutResult(reward_result=RewardResult(reward=reward)),
        aborted=aborted,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_env():
    """Mock Environment with async reset/rollout/cleanup."""
    env = MagicMock(spec=Environment)
    env.reset = AsyncMock()
    env.rollout = AsyncMock()
    env.cleanup = AsyncMock()
    return env


@pytest.fixture
def runner():
    """Click CliRunner for CLI tests."""
    return CliRunner()
