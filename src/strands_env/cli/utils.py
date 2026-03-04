# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utility functions."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import click

from strands_env.core.models import ModelFactory, bedrock_model_factory, kimi_model_factory, sglang_model_factory

from .config import EnvConfig, ModelConfig

if TYPE_CHECKING:
    from strands_sglang.tool_parsers import ToolParser

    from strands_env.eval import AsyncEnvFactory, Evaluator

#: Type for the create_env_factory function exported by hook files.
EnvFactoryCreator = Callable[[ModelFactory, EnvConfig], "AsyncEnvFactory"]

#: Type for evaluator class.
EvaluatorClass = type["Evaluator"]


# ---------------------------------------------------------------------------
# Hook Loading
# ---------------------------------------------------------------------------


def _load_hook_module(path: Path, hook_name: str) -> ModuleType:
    """Load a Python module from a file path.

    Args:
        path: Path to the Python file.
        hook_name: Name for the module (used in error messages).

    Returns:
        The loaded module.

    Raises:
        click.ClickException: If the file cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(hook_name, path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load {hook_name} file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_env_hook(path: Path) -> EnvFactoryCreator:
    """Load environment hook file and return create_env_factory function.

    The hook file must export a `create_env_factory(model_factory, env_config)` function.

    Args:
        path: Path to the Python hook file.

    Returns:
        The create_env_factory function from the hook file.

    Raises:
        click.ClickException: If the file cannot be loaded or doesn't export the function.
    """
    module = _load_hook_module(path, "env_hook")

    if not hasattr(module, "create_env_factory"):
        raise click.ClickException(
            "Hook file must export 'create_env_factory(model_factory, env_config)' function.\n"
            "Example:\n"
            "  def create_env_factory(model_factory, env_config):\n"
            "      async def env_factory(action):\n"
            "          return MyEnv(\n"
            "              model_factory=model_factory,\n"
            "              system_prompt=env_config.system_prompt,\n"
            "              max_tool_iters=env_config.max_tool_iters,\n"
            "              max_tool_calls=env_config.max_tool_calls,\n"
            "          )\n"
            "      return env_factory"
        )

    return cast(EnvFactoryCreator, module.create_env_factory)


def load_evaluator_hook(path: Path) -> EvaluatorClass:
    """Load evaluator hook file and return the Evaluator class.

    The hook file must export an `EvaluatorClass` that extends `Evaluator`.

    Args:
        path: Path to the Python hook file.

    Returns:
        The Evaluator subclass from the hook file.

    Raises:
        click.ClickException: If the file cannot be loaded or doesn't export the class.
    """
    from strands_env.eval import Evaluator

    module = _load_hook_module(path, "evaluator_hook")

    if not hasattr(module, "EvaluatorClass"):
        raise click.ClickException(
            "Evaluator hook file must export 'EvaluatorClass' (an Evaluator subclass).\n"
            "Example:\n"
            "  from strands_env.eval import Evaluator\n"
            "\n"
            "  class MyEvaluator(Evaluator):\n"
            "      benchmark_name = 'my-benchmark'\n"
            "\n"
            "      def load_dataset(self):\n"
            "          ...\n"
            "\n"
            "  EvaluatorClass = MyEvaluator"
        )

    evaluator_cls = module.EvaluatorClass
    if not isinstance(evaluator_cls, type) or not issubclass(evaluator_cls, Evaluator):
        raise click.ClickException("EvaluatorClass must be a subclass of Evaluator")

    return evaluator_cls


def load_tool_parser(tool_parser_arg: str | None) -> ToolParser | None:
    """Load tool parser from name or hook file path.

    Args:
        tool_parser_arg: Either a parser name (e.g., "hermes", "qwen_xml") or path to hook file.

    Returns:
        ToolParser instance, or None if not specified.

    Raises:
        click.ClickException: If the parser name is unknown or hook file is invalid.
    """
    if tool_parser_arg is None:
        return None

    # Check if it's a file path
    path = Path(tool_parser_arg)
    if path.exists() and path.is_file():
        return _load_tool_parser_hook(path)

    # Otherwise treat as parser name
    from strands_sglang.tool_parsers import get_tool_parser

    try:
        return get_tool_parser(tool_parser_arg)
    except KeyError as e:
        raise click.ClickException(str(e)) from e


def _load_tool_parser_hook(path: Path) -> ToolParser:
    """Load tool parser from hook file.

    The hook file must export either:
    - `tool_parser`: A ToolParser instance
    - `ToolParserClass`: A ToolParser subclass (will be instantiated)

    Args:
        path: Path to the Python hook file.

    Returns:
        ToolParser instance from the hook file.

    Raises:
        click.ClickException: If the file cannot be loaded or doesn't export the parser.
    """
    from strands_sglang.tool_parsers import ToolParser

    module = _load_hook_module(path, "tool_parser_hook")

    # Check for tool_parser instance first
    if hasattr(module, "tool_parser"):
        parser = module.tool_parser
        if not isinstance(parser, ToolParser):
            raise click.ClickException("'tool_parser' must be a ToolParser instance")
        return parser

    # Check for ToolParserClass
    if hasattr(module, "ToolParserClass"):
        parser_cls = module.ToolParserClass
        if not isinstance(parser_cls, type) or not issubclass(parser_cls, ToolParser):
            raise click.ClickException("'ToolParserClass' must be a ToolParser subclass")
        return parser_cls()

    raise click.ClickException(
        "Tool parser hook file must export 'tool_parser' (instance) or 'ToolParserClass' (subclass).\n"
        "Example:\n"
        "  from strands_sglang.tool_parsers import ToolParser, ToolParseResult\n"
        "\n"
        "  class MyToolParser(ToolParser):\n"
        "      def parse(self, text: str) -> list[ToolParseResult]:\n"
        "          ...\n"
        "\n"
        "  tool_parser = MyToolParser()\n"
        "  # OR\n"
        "  ToolParserClass = MyToolParser"
    )


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


def build_model_factory(config: ModelConfig, max_concurrency: int) -> ModelFactory:
    """Build a ModelFactory from ModelConfig.

    Args:
        config: Model configuration.
        max_concurrency: Max concurrent connections (for SGLang client pooling).

    Returns:
        ModelFactory callable.
    """
    sampling = config.sampling.to_dict()

    if config.backend == "sglang":
        return _build_sglang_model_factory(config, max_concurrency, sampling)
    elif config.backend == "bedrock":
        return _build_bedrock_model_factory(config, sampling)
    elif config.backend == "kimi":
        return _build_kimi_model_factory(config, sampling)
    else:
        raise click.ClickException(f"Unknown backend: {config.backend}")


def _build_sglang_model_factory(config: ModelConfig, max_concurrency: int, sampling: dict) -> ModelFactory:
    """Build SGLang model factory."""
    from strands_sglang import get_client, get_tokenizer

    from strands_env.utils.sglang import check_server_health, get_model_id

    # Check server health before proceeding
    try:
        check_server_health(config.base_url)
    except ConnectionError as e:
        raise click.ClickException(str(e)) from e

    client = get_client(config.base_url, max_connections=max_concurrency)

    # Resolve and backfill model_id/tokenizer_path for reproducibility
    if not config.model_id:
        config.model_id = get_model_id(config.base_url)
    if not config.tokenizer_path:
        config.tokenizer_path = config.model_id

    tokenizer = get_tokenizer(config.tokenizer_path)
    tool_parser = load_tool_parser(config.tool_parser)

    return sglang_model_factory(
        client=client,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
        sampling_params=sampling,
    )


def _build_bedrock_model_factory(config: ModelConfig, sampling: dict) -> ModelFactory:
    """Build Bedrock model factory."""
    from strands_env.utils.aws import get_session

    if not config.model_id:
        raise click.ClickException("--model-id is required for Bedrock backend")

    boto_session = get_session(
        region=config.region or "us-east-1",
        profile_name=config.profile_name,
        role_arn=config.role_arn,
    )

    return bedrock_model_factory(model_id=config.model_id, boto_session=boto_session, sampling_params=sampling)


def _build_kimi_model_factory(config: ModelConfig, sampling: dict) -> ModelFactory:
    """Build Kimi (Moonshot AI) model factory via LiteLLM."""
    import os

    if not os.getenv("MOONSHOT_API_KEY"):
        raise click.ClickException("MOONSHOT_API_KEY environment variable is required for Kimi backend")

    return kimi_model_factory(
        model_id=config.model_id or "moonshot/kimi-k2.5",
        sampling_params=sampling,
    )
