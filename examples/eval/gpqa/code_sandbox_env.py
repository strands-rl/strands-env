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

"""Example environment hook for GPQA evaluation with CodeSandboxEnv."""

from strands_env.core.models import build_model_factory
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.eval.benchmarks.gpqa import GPQAReward
from strands_env.tools import CodeInterpreterQuotas
from strands_env.utils.aws import get_client

QUOTAS = CodeInterpreterQuotas()


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for AIME evaluation with Python execution."""
    model_factory = build_model_factory(model_config)
    reward_fn = GPQAReward()
    client = get_client(service_name="bedrock-agentcore", role_arn=env_config.get("agentcore_role_arn"))

    async def env_factory(_action):
        return CodeSandboxEnv(
            model_factory=model_factory, reward_fn=reward_fn, mode="code", client=client, quotas=QUOTAS, **env_config
        )

    return env_factory

