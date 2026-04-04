# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example environment hook for MCP-Atlas evaluation."""

from __future__ import annotations

import httpx

from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.environments.mcp_atlas import MCPAtlasEnvironment, MCPAtlasRewardFunction
from strands_env.utils.aws import get_session


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for MCP-Atlas benchmark tasks."""
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(region="us-west-2", profile_name=profile_name)
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    max_judge_retries = env_config.get("max_judge_retries", 3)

    docker_url = env_config.get("docker_url", MCPAtlasEnvironment.DEFAULT_DOCKER_URL)
    http_client: httpx.AsyncClient | None = None

    def _get_client() -> httpx.AsyncClient:
        nonlocal http_client
        if http_client is None or http_client.is_closed:
            http_client = MCPAtlasEnvironment.create_client(base_url=docker_url)
        return http_client

    async def env_factory(action):
        ctx = action.task_context
        # Each env gets its own reward_fn to avoid concurrent tasks overwriting
        # _current_claim / _response on a shared instance.
        reward_fn = MCPAtlasRewardFunction(judge_model=judge_models, max_model_retries=max_judge_retries)
        return MCPAtlasEnvironment(
            model_factory=model_factory,
            http_client=_get_client(),
            reward_fn=reward_fn,
            enabled_tools=ctx.enabled_tools,
            **env_config,
        )

    return env_factory
