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

"""Example environment hook for SimpleQA-Verified evaluation with a chat-only environment (no tools).

The judge model uses Bedrock by default. Set JUDGE_MODEL_ID to override.
"""

import os

from strands_env.core import Environment
from strands_env.core.models import ModelFactory, bedrock_model_factory
from strands_env.eval.benchmarks.simpleqa_verified import SimpleQAReward
from strands_env.utils.aws import get_session

#: Default judge model for grading answers.
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")


def create_env_factory(model_factory: ModelFactory, **env_config):
    """Create env_factory for chat-only SimpleQA-Verified evaluation."""
    boto_session = get_session(region="us-west-2")
    judge_model_factory = bedrock_model_factory(
        model_id=JUDGE_MODEL_ID,
        boto_session=boto_session,
        sampling_params={"max_new_tokens": 1024},
    )
    reward_fn = SimpleQAReward(judge_model=judge_model_factory())

    async def env_factory(_action):
        return Environment(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
