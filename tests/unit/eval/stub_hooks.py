from unittest.mock import AsyncMock, MagicMock

from strands_env.core import RewardResult, RolloutResult, Task
from strands_env.eval import Evaluator


class EvaluatorClass(Evaluator):
    """One prompt, always solved; enough for the CLI to run end to end."""

    benchmark_name = "stub"

    def load_dataset(self):
        return [Task(id="p1", message="q1", ground_truth="42")]


def create_env_factory(model_config: dict, **env_config):
    async def env_factory():
        env = MagicMock()
        env.rollout = AsyncMock(return_value=RolloutResult(reward_result=RewardResult(reward=1.0)))
        return env

    return env_factory
