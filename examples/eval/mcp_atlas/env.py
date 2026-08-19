"""Example environment hook for MCP-Atlas evaluation."""

from __future__ import annotations

from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.environments.mcp_atlas import MCPAtlasEnv, MCPAtlasReward
from strands_env.utils.aws import get_session


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for MCP-Atlas benchmark tasks."""
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(region_name="us-west-2", profile_name=profile_name)
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    docker_url = env_config.get("docker_url", MCPAtlasEnv.DEFAULT_DOCKER_URL)
    http_client = MCPAtlasEnv.create_client(base_url=docker_url)

    async def env_factory():
        # Each env gets its own reward_fn to avoid concurrent tasks overwriting
        # _current_claim / _response on a shared instance.
        reward_fn = MCPAtlasReward(judge_model=judge_models)
        return MCPAtlasEnv(
            model_factory=model_factory,
            http_client=http_client,
            reward_fn=reward_fn,
            **env_config,
        )

    env_factory.http_client = http_client  # type: ignore[attr-defined]
    return env_factory
