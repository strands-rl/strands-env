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

"""Example environment hook for HLE-Verified evaluation with a chat-only environment (no tools)."""

from strands_env.core import Environment
from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.eval.benchmarks.hle_verified import HLEReward
from strands_env.utils.aws import get_session


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for chat-only HLE-Verified evaluation.

    Works for both `hle-verified-gold` (includes multimodal samples) and
    `hle-verified-gold-text` (text-only). For the multimodal variant, make sure
    the backend model supports image content blocks.
    """
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(region="us-west-2", profile_name=profile_name)
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                # HLE judgment emits extracted_final_answer + reasoning + yes/no, so give it more
                # headroom than the single-word SimpleQA judge. Live probes over Gold samples topped
                # out around ~170 output tokens, so 2048 is comfortably above worst case.
                sampling_params={"max_new_tokens": 2048},
            )()
        )
    reward_fn = HLEReward(judge_model=judge_models, max_model_retries=env_config.get("max_judge_retries", 3))

    async def env_factory(_action):
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
