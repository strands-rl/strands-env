"""Example environment hook for math reasoning evaluation with `CalculatorEnv`."""

from strands_env.core.models import build_model_factory
from strands_env.environments.calculator import CalculatorEnv
from strands_env.environments.calculator.reward import MathVerifyReward


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `CalculatorEnv`."""
    model_factory = build_model_factory(model_config)
    reward_fn = MathVerifyReward()

    async def env_factory():
        return CalculatorEnv(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
