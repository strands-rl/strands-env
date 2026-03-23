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

"""Environment hook for terminal-bench-2 evaluation on AWS EKS/Fargate via harbor-aws.

Usage:
    strands-env eval run terminal-bench-2 \
        --env examples.eval.terminal_bench.terminal_bench_eks_env \
        --backend bedrock \
        --model-id moonshotai.kimi-k2.5 \
        --region us-east-1 \
        --max-concurrency 89
"""

from __future__ import annotations

import os

from strands_env.core.models import ModelFactory
from strands_env.core.types import Action
from strands_env.environments.terminal_bench import TerminalBenchEnv
from strands_env.environments.terminal_bench.env import EKSBackendConfig

# Configure via environment variables or edit defaults here.
EKS_CONFIG = EKSBackendConfig(
    stack_name=os.environ.get("HARBOR_STACK_NAME", "harbor-aws"),
    region=os.environ.get("HARBOR_REGION", "us-east-1"),
    ecr_cache=os.environ.get("HARBOR_ECR_CACHE", "true").lower() == "true",
    role_arn=os.environ.get("HARBOR_ROLE_ARN"),
)


def create_env_factory(model_factory: ModelFactory, **env_config):
    """Create env_factory for TerminalBenchEnv on EKS/Fargate."""

    async def env_factory(action: Action) -> TerminalBenchEnv:
        """Create a new TerminalBenchEnv backed by an EKS Fargate pod."""
        ctx = action.task_context
        return TerminalBenchEnv(
            model_factory=model_factory,
            backend="eks",
            eks_backend_config=EKS_CONFIG,
            **ctx.config,
            **env_config,
        )

    return env_factory
