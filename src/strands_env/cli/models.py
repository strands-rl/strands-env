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
from typing import Any, Literal

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
class ModelConfig:
    """Serializable model configuration for CLI and distributed evaluation."""

    backend: Literal["sglang", "bedrock", "kimi"] = "sglang"

    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None
    tool_parser: str | None = None
    max_connections: int = 1000

    # Bedrock / model identifier
    model_id: str | None = None
    region: str = "us-west-2"
    profile_name: str | None = None
    role_arn: str | None = None

    # Sampling
    sampling_params: dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 16384})

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return dataclasses.asdict(self)


def build_model_factory(config: ModelConfig | dict[str, Any]) -> ModelFactory:
    """Build a `ModelFactory` from a `ModelConfig` or a serialized config dict.

    Args:
        config: Model configuration (dataclass or dict from `ModelConfig.to_dict()`).
    """
    if isinstance(config, dict):
        config = ModelConfig(**config)

    match config.backend:
        case "sglang":
            check_server_health(config.base_url)
            client = get_client(config.base_url, max_connections=config.max_connections)
            config.model_id = config.model_id or get_model_id(config.base_url)
            config.tokenizer_path = config.tokenizer_path or config.model_id
            tool_parser = config.tool_parser or "hermes"
            return sglang_model_factory(
                client=client,
                tokenizer=get_tokenizer(config.tokenizer_path),
                tool_parser=get_tool_parser(tool_parser),
                sampling_params=config.sampling_params,
            )
        case "bedrock":
            config.model_id = config.model_id or "us.anthropic.claude-sonnet-4-20250514-v1:0"
            boto_session = get_session(
                region=config.region,
                profile_name=config.profile_name,
                role_arn=config.role_arn,
            )
            return bedrock_model_factory(
                model_id=config.model_id, boto_session=boto_session, sampling_params=config.sampling_params
            )
        case "kimi":
            return kimi_model_factory(
                model_id=config.model_id or "moonshot/kimi-k2.5",
                sampling_params=config.sampling_params,
            )
