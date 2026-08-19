"""Example environment hook for IFEval with a chat-only environment (no tools)."""

from strands_env.core import Environment
from strands_env.core.models import build_model_factory
from strands_env.eval.benchmarks.ifeval import IFEvalReward


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for chat-only IFEval evaluation."""
    model_factory = build_model_factory(model_config)
    reward_fn = IFEvalReward(metric=env_config.pop("ifeval_metric", "prompt_level_strict_acc"))

    async def env_factory():
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
