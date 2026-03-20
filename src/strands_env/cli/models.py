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

"""Model configuration and factory building for CLI."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from strands_sglang import get_client, get_tokenizer
from strands_sglang.tool_parsers import get_tool_parser

from strands_env.core.models import (
    ModelFactory,
    bedrock_model_factory,
    kimi_model_factory,
    sglang_model_factory,
)
from strands_env.utils.aws import get_session
from strands_env.utils.sglang import check_server_health, get_model_id


@dataclass
class SamplingConfig:
    """Sampling parameters for model generation."""

    temperature: float | None = None
    max_new_tokens: int = 16384
    top_p: float | None = None
    top_k: int | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding `None` values so the model uses its own defaults."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclass
class ModelConfig:
    """Model configuration."""

    backend: Literal["sglang", "bedrock", "kimi"] = "sglang"

    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None  # Auto-detected if None
    tool_parser: str | None = None  # Parser name or path to hook file

    # Bedrock
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    region: str = "us-west-2"
    profile_name: str | None = None  # AWS profile name
    role_arn: str | None = None  # For role assumption

    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = dataclasses.asdict(self)
        d["sampling"] = self.sampling.to_dict()
        return d


def build_model_factory(config: ModelConfig, max_concurrency: int) -> ModelFactory:
    """Build a `ModelFactory` from `ModelConfig`.

    Args:
        config: Model configuration.
        max_concurrency: Max concurrent connections (for SGLang client pooling).
    """
    sampling = config.sampling.to_dict()

    match config.backend:
        case "sglang":
            check_server_health(config.base_url)
            client = get_client(config.base_url, max_connections=max_concurrency)
            config.model_id = get_model_id(config.base_url) if not config.model_id else config.model_id
            config.tokenizer_path = config.model_id if not config.tokenizer_path else config.tokenizer_path
            return sglang_model_factory(
                client=client,
                tokenizer=get_tokenizer(config.tokenizer_path),
                tool_parser=get_tool_parser(config.tool_parser) if config.tool_parser else get_tool_parser("hermes"),
                sampling_params=sampling,
            )
        case "bedrock":
            boto_session = get_session(
                region=config.region or "us-east-1",
                profile_name=config.profile_name,
                role_arn=config.role_arn,
            )
            return bedrock_model_factory(model_id=config.model_id, boto_session=boto_session, sampling_params=sampling)

        case "kimi":
            return kimi_model_factory(
                model_id=config.model_id or "moonshot/kimi-k2.5",
                sampling_params=sampling,
            )
        # TODO: add more backends here
