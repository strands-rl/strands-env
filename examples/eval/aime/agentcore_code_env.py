"""Example environment hook for math reasoning evaluation with `AgentCoreCodeEnv`."""

from strands_env.core.models import build_model_factory
from strands_env.environments.agentcore_code import AgentCoreCodeEnv
from strands_env.environments.agentcore_code.quotas import CodeInterpreterQuotas
from strands_env.environments.math.reward import MathVerifyReward
from strands_env.utils.aws import get_client

QUOTAS = CodeInterpreterQuotas()


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for AIME evaluation with Python execution."""
    model_factory = build_model_factory(model_config)
    reward_fn = MathVerifyReward()
    client = get_client(service_name="bedrock-agentcore", role_arn=env_config.get("agentcore_role_arn"))

    async def env_factory():
        return AgentCoreCodeEnv(
            model_factory=model_factory, reward_fn=reward_fn, mode="code", client=client, quotas=QUOTAS, **env_config
        )

    return env_factory
