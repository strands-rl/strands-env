from strands_env.core import Environment
from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.eval.benchmarks.sealqa import SealQAReward
from strands_env.utils.aws import get_session


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for chat-only SealQA evaluation."""
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(
            region_name="us-west-2", profile_name=profile_name, role_arn=env_config.get("judge_model_role_arn")
        )
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    reward_fn = SealQAReward(judge_model=judge_models)

    async def env_factory():
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
