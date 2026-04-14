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

"""Model factory functions for supported backends.

Each function returns a `ModelFactory` (zero-arg callable that creates a fresh
`Model` instance) for use with `Environment`::

    from strands_sglang import SGLangClient
    from strands_env.core.models import sglang_model_factory

    client = SGLangClient("http://localhost:30000")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    env = Environment(
        model_factory=sglang_model_factory(tokenizer=tokenizer, client=client),
    )

Users can easily create their own model factories by implementing the `ModelFactory` type.

Example:
    >>> from strands_env.core.models import ModelFactory
    >>> def my_model_factory() -> ModelFactory:
    >>>     return lambda: MyModel()
    >>>
    >>> env = Environment(model_factory=my_model_factory())
    >>>
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import boto3
import botocore.config
import httpx
from strands.models import Model
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel
from strands.types.content import Messages
from strands_sglang import SGLangClient, SGLangModel, get_client, get_tokenizer
from strands_sglang.tool_parsers import HermesToolParser, ToolParser, get_tool_parser
from transformers import PreTrainedTokenizerBase

from strands_env.utils.aws import get_session
from strands_env.utils.decorators import requires_env
from strands_env.utils.sglang import check_server_health, get_model_id

#: Factory that produces a fresh `Model` per step (for concurrent step isolation).
ModelFactory = Callable[[], Model]

# Other parameters like temperature and top_p will be set to model's default values if provided
DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 16384}

# ---------------------------------------------------------------------------
# SGLang Model
# ---------------------------------------------------------------------------


def sglang_model_factory(
    *,
    client: SGLangClient,
    tokenizer: PreTrainedTokenizerBase,
    tool_parser: ToolParser | None = None,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    return_logprob: bool = True,
    return_routed_experts: bool = False,
    enable_thinking: bool = True,
) -> ModelFactory:
    """Return a factory that creates `SGLangModel` instances.

    Args:
        client: `SGLangClient` for HTTP communication with the SGLang server.
        tokenizer: HuggingFace tokenizer for chat template and tokenization.
        tool_parser: Tool parser for extracting tool calls from model output. Defaults to `HermesToolParser`.
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
        return_logprob: Whether to return logprobs for each token.
        return_routed_experts: Whether to return MoE routed expert indices for routing replay.
        enable_thinking: Enable thinking mode for models whose chat template supports it.
    """
    if tool_parser is None:
        tool_parser = HermesToolParser()

    return lambda: SGLangModel(
        client=client,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        sampling_params=sampling_params,
        return_logprob=return_logprob,
        return_routed_experts=return_routed_experts,
        enable_thinking=enable_thinking,
    )


# ---------------------------------------------------------------------------
# Bedrock Model
# ---------------------------------------------------------------------------


DEFAULT_BOTO_CLIENT_CONFIG = botocore.config.Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    max_pool_connections=100,
    connect_timeout=5.0,
    read_timeout=600.0,
)


def bedrock_model_factory(
    *,
    model_id: str,
    boto_session: boto3.Session,
    boto_client_config: botocore.config.Config = DEFAULT_BOTO_CLIENT_CONFIG,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
) -> ModelFactory:
    """Return a factory that creates `BedrockModel` instances.

    Args:
        model_id: Bedrock model ID (e.g. `"us.anthropic.claude-sonnet-4-20250514-v1:0"`).
        boto_session: Boto3 session for AWS credentials.
        boto_client_config: Botocore client configuration.
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).

    Notes:
        - A single boto3 client (thread-safe) is created once from the session and
        shared across all model instances.  `BedrockModel` doesn't accept a pre-built client,
        so we extract it from a pilot instance and override `model.client` on each subsequent one.
        - The principle of operation is "one boto3 session, one boto3 client".
        - `max_new_tokens` in `sampling_params` is remapped to `max_tokens` for the Bedrock API.
    """
    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    model_kwargs = dict(
        model_id=model_id,
        boto_session=boto_session,
        boto_client_config=boto_client_config,
        streaming=False,
        **sampling_params,
    )

    # Build one model to extract a properly configured, thread-safe client.
    shared_client = BedrockModel(**model_kwargs).client

    def factory() -> BedrockModel:
        model = BedrockModel(**model_kwargs)
        model.client = shared_client
        return model

    return factory


# ---------------------------------------------------------------------------
# OpenAI Model
# ---------------------------------------------------------------------------

# OpenAI client arguments for SGLang server
DEFAULT_OPENAI_CLIENT_ARGS = {
    "api_key": "EMPTY",
    "base_url": "http://localhost:30000/v1",
    "timeout": httpx.Timeout(timeout=600.0, connect=5.0),
    "max_retries": 5,
}


def openai_model_factory(
    *,
    model_id: str,
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    client_args: dict[str, Any] = DEFAULT_OPENAI_CLIENT_ARGS,
) -> ModelFactory:
    """Return a factory that creates `OpenAIModel` instances.

    Args:
        model_id: OpenAI model ID (e.g. `"gpt-4o"`).
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
        client_args: Arguments for the OpenAI client (e.g. `{"api_key": "...", "base_url": "..."}`).

    Notes:
        `max_new_tokens` in `sampling_params` is remapped to `max_tokens` for the OpenAI API.
    """
    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    return lambda: OpenAIModel(
        model_id=model_id,
        params=sampling_params,
        client_args=client_args,
    )


# ---------------------------------------------------------------------------
# Kimi Model (Moonshot AI — via LiteLLM)
# ---------------------------------------------------------------------------


def _get_kimi_model_class() -> type:
    """Return a LiteLLMModel subclass that preserves reasoning_content for Moonshot.

    Notes:
        - Both `OpenAIModel` and `LiteLLMModel` strip `reasoningContent` in `_format_regular_messages`,
        but Moonshot requires it back as a top-level `reasoning_content` field in multi-turn messages.
    """
    from strands.models.litellm import LiteLLMModel

    class KimiModel(LiteLLMModel):
        @classmethod
        def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
            # Extract reasoning text before super() strips reasoningContent blocks
            reasoning_map: dict[int, str] = {}
            for i, message in enumerate(messages):
                parts = [
                    content["reasoningContent"].get("reasoningText", {}).get("text", "")
                    for content in message["content"]
                    if "reasoningContent" in content
                ]
                if any(parts):
                    reasoning_map[i] = "".join(parts)

            # Delegate to parent (strips reasoningContent, formats toolUse/toolResult)
            formatted_messages = super()._format_regular_messages(messages, **kwargs)

            # Re-inject reasoning_content into the corresponding formatted messages.
            # super() emits one primary message per original message (same role),
            # plus extra "tool" role messages for toolResult blocks — skip those.
            orig_idx = 0
            for fmt_msg in formatted_messages:
                if fmt_msg.get("role") == "tool":
                    continue
                if orig_idx in reasoning_map:
                    fmt_msg["reasoning_content"] = reasoning_map[orig_idx]
                orig_idx += 1

            return formatted_messages

    return KimiModel


@requires_env("MOONSHOT_API_KEY")
def kimi_model_factory(
    *,
    model_id: str = "moonshot/kimi-k2.5",
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    client_args: dict[str, Any] | None = None,
) -> ModelFactory:
    """Return a factory that creates `KimiModel` instances for Moonshot AI.

    Args:
        model_id: LiteLLM model ID with `moonshot/` prefix (default `"moonshot/kimi-k2.5"`).
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 4096}`).
        client_args: Arguments for the LiteLLM client.

    Notes:
        - Requires `MOONSHOT_API_KEY` environment variable.
        - `max_new_tokens` in `sampling_params` is remapped to `max_tokens` for the LiteLLM API.
    """
    kimi_model_cls = _get_kimi_model_class()

    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_tokens"] = sampling_params.pop("max_new_tokens")

    return lambda: kimi_model_cls(
        model_id=model_id,
        params=sampling_params,
        client_args=client_args,
    )


# ---------------------------------------------------------------------------
# Model Configuration and Factory (mainly for CLI)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Serializable model configuration."""

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

    This is the config-driven path for creating model factories, used by
    eval hooks and Ray actors. For programmatic use with pre-built objects
    (clients, tokenizers), use the individual factory functions directly.

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
        case _:
            raise ValueError(f"Unsupported backend for ModelConfig: {config.backend!r}")
