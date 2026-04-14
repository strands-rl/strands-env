# Copyright 2025-2026 Strands RL Contributors
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

"""Example environment hook for BrowseComp evaluation with a chat-only environment (no tools)."""

from strands_env.core import Environment
from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.eval.benchmarks.browsecomp import BrowseCompReward
from strands_env.utils.aws import get_session


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for chat-only BrowseComp evaluation."""
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(
            region="us-west-2", profile_name=profile_name, role_arn=env_config.get("judge_model_role_arn", None)
        )
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    reward_fn = BrowseCompReward(judge_model=judge_models, max_model_retries=env_config.get("max_judge_retries", 3))

    async def env_factory(_action):
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
