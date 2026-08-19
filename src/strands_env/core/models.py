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
from strands.models.openai_responses import OpenAIResponsesModel
from strands_sglang import SGLangClient, SGLangModel, get_client, get_tokenizer
from strands_sglang.tool_parsers import HermesToolParser, ToolParser, get_tool_parser
from transformers import PreTrainedTokenizerBase

from strands_env.utils.aws import get_session

#: Factory that produces a fresh `Model` per rollout (for concurrent rollout isolation).
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
# SGLang server helpers
# ---------------------------------------------------------------------------


def check_server_health(base_url: str, timeout: float = 5.0) -> None:
    """Check if the SGLang server is reachable.

    Notes:
        Sync convenience using httpx. For async runtime use, see
        `SGLangClient.health()` which uses aiohttp.
    """
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise ConnectionError(f"SGLang server at {base_url} is not reachable: {e}") from e


def get_model_id(base_url: str, timeout: float = 5.0) -> str:
    """Get the model's HF identifier from the SGLang server via `/model_info`."""
    response = httpx.get(f"{base_url}/model_info", timeout=timeout)
    response.raise_for_status()
    info = response.json()
    model_id = info.get("tokenizer_path") or info.get("model_path")
    if not model_id:
        raise RuntimeError(f"No tokenizer_path/model_path at {base_url}/model_info")
    return str(model_id)


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
# Bedrock Mantle Model (GPT via OpenAI Responses API on AWS)
# ---------------------------------------------------------------------------


def bedrock_mantle_model_factory(
    *,
    model_id: str,
    region: str = "us-east-2",
    sampling_params: dict[str, Any] = DEFAULT_SAMPLING_PARAMS,
    reasoning: dict[str, Any] | None = None,
) -> ModelFactory:
    """Return a factory that creates `OpenAIResponsesModel` for GPT models via Bedrock Mantle.

    Args:
        model_id: Bedrock Mantle model ID (e.g. `"openai.gpt-5.4-2026-03-05"`).
        region: AWS region hosting Bedrock Mantle (default `"us-east-2"`).
        sampling_params: Sampling parameters for the model (e.g. `{"max_new_tokens": 16384}`).
        reasoning: Reasoning configuration (e.g. `{"effort": "high"}`).

    Notes:
        - Routing is delegated to the SDK's `bedrock_mantle_config`: it derives the regional
          base URL (`openai.gpt-5.*` → `/openai/v1`, other models → `/v1`) and mints a fresh
          SigV4 token per request via `aws_bedrock_token_generator`, so long rollouts never
          outlive the bearer token.
        - Leaves server-side conversation state at the SDK default (`stateful=False`), matching
          the other stateless backends: the full transcript is sent each turn and `agent.messages`
          stays intact for `Environment` observation capture. (With `stateful=True` the SDK clears
          `agent.messages` for server-managed conversations, which would discard the trajectory.)
        - Reasoning tokens are NOT retained across turns: the SDK filters `reasoningContent` from
          Responses API requests because the round-trip metadata (`encrypted_content`) has nowhere
          to live on the message, so each turn reasons fresh. Fine for single-turn eval; multi-turn
          reasoning continuity awaits upstream support.
          See https://github.com/strands-agents/harness-sdk/issues/2014.
    """
    sampling_params = dict(sampling_params)
    if "max_new_tokens" in sampling_params:
        sampling_params["max_output_tokens"] = sampling_params.pop("max_new_tokens")

    params: dict[str, Any] = {**sampling_params}
    if reasoning:
        params["reasoning"] = reasoning

    def factory() -> OpenAIResponsesModel:
        return OpenAIResponsesModel(
            model_id=model_id,
            params=params,
            bedrock_mantle_config={"region": region},
        )

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
# Model Configuration and Factory (mainly for CLI)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Serializable model configuration."""

    backend: Literal["sglang", "bedrock", "bedrock-mantle"] = "sglang"

    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None
    tool_parser: str | None = None
    max_connections: int = 1000

    # Bedrock / model identifier
    model_id: str | None = None
    region_name: str | None = None
    profile_name: str | None = None
    role_arn: str | None = None

    # Bedrock Mantle (GPT via OpenAI Responses API): reasoning config, e.g. {"effort": "high"}.
    reasoning: dict[str, Any] | None = None

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
                region_name=config.region_name,
                profile_name=config.profile_name,
                role_arn=config.role_arn,
            )
            return bedrock_model_factory(
                model_id=config.model_id, boto_session=boto_session, sampling_params=config.sampling_params
            )
        case "bedrock-mantle":
            if not config.model_id:
                raise ValueError("bedrock-mantle backend requires --model-id (e.g. 'openai.gpt-5.4-2026-03-05').")
            return bedrock_mantle_model_factory(
                model_id=config.model_id,
                region=config.region_name or "us-east-2",
                sampling_params=config.sampling_params,
                reasoning=config.reasoning,
            )
        case _:
            raise ValueError(f"Unsupported backend for ModelConfig: {config.backend!r}")
