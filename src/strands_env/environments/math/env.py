"""Math environment: no tools, symbolic-equivalence reward."""

from pathlib import Path
from typing import Unpack, override

from strands_env.core import Environment, EnvironmentConfig, ModelFactory, RewardFunction
from strands_env.environments.math.reward import MathVerifyReward


class MathEnv(Environment):
    """Chat-only math environment, scored by `MathVerifyReward` on the boxed answer."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[EnvironmentConfig],
    ):
        super().__init__(model_factory=model_factory, reward_fn=reward_fn or MathVerifyReward(), **config)

    @override
    def get_tools(self) -> list:
        """No tools: the model reasons in text and boxes its answer."""
        return []
