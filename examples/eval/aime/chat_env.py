from strands_env.core import Environment
from strands_env.core.models import build_model_factory
from strands_env.environments.math.reward import MathVerifyReward


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for chat-only AIME evaluation."""
    model_factory = build_model_factory(model_config)
    reward_fn = MathVerifyReward()

    async def env_factory():
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
