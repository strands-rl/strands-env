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
class SamplingParams:
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
    max_connections: int = 1000  # Max concurrent connections (for SGLang client pooling)

    # Bedrock / model identifier (auto-detected for SGLang; defaults to Claude Sonnet for Bedrock)
    model_id: str | None = None
    region: str = "us-west-2"
    profile_name: str | None = None  # AWS profile name
    role_arn: str | None = None  # For role assumption

    # Sampling
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = dataclasses.asdict(self)
        d["sampling"] = self.sampling_params.to_dict()
        return d


def build_model_factory(config: ModelConfig) -> ModelFactory:
    """Build a `ModelFactory` from `ModelConfig`.

    Args:
        config: Model configuration.
    """
    sampling_params = config.sampling_params.to_dict()

    match config.backend:
        case "sglang":
            check_server_health(config.base_url)
            client = get_client(config.base_url, max_connections=config.max_connections)
            config.model_id = config.model_id or get_model_id(config.base_url)
            config.tokenizer_path = config.tokenizer_path or config.model_id
            return sglang_model_factory(
                client=client,
                tokenizer=get_tokenizer(config.tokenizer_path),
                tool_parser=get_tool_parser(config.tool_parser) if config.tool_parser else get_tool_parser("hermes"),
                sampling_params=sampling_params,
            )
        case "bedrock":
            config.model_id = config.model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
            boto_session = get_session(
                region=config.region or "us-east-1",
                profile_name=config.profile_name,
                role_arn=config.role_arn,
            )
            return bedrock_model_factory(
                model_id=config.model_id, boto_session=boto_session, sampling_params=sampling_params
            )

        case "kimi":
            return kimi_model_factory(
                model_id=config.model_id or "moonshot/kimi-k2.5",
                sampling_params=sampling_params,
            )
        # TODO: add more backends here
