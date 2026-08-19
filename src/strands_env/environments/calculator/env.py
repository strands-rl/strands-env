"""Math environment: a calculator tool with symbolic-equivalence reward."""

from pathlib import Path
from typing import Unpack, override

from strands_tools.calculator import calculator

from strands_env.core import Environment, EnvironmentConfig, ModelFactory, RewardFunction
from strands_env.environments.calculator.reward import MathVerifyReward


class CalculatorEnv(Environment):
    """Math environment with a `calculator` tool and `MathVerifyReward` by default."""

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
        """Return the `calculator` tool from `strands_tools`."""
        return [calculator]
