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

"""Evaluation CLI commands."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import click

from strands_env.eval import get_benchmark, list_benchmarks, list_unavailable_benchmarks

from .config import EnvConfig, EvalConfig, ModelConfig, SamplingConfig
from .utils import build_model_factory, load_env_hook, load_evaluator_hook

logger = logging.getLogger(__name__)


@click.group("eval")
def eval_group() -> None:
    """Benchmark evaluation commands."""
    pass


@eval_group.command("list")
def list_cmd() -> None:
    """List registered benchmarks."""
    benchmarks = list_benchmarks()
    unavailable = list_unavailable_benchmarks()

    click.echo("Benchmarks:")
    if benchmarks:
        for name in benchmarks:
            click.echo(f"  - {name}")
    else:
        click.echo("  (none registered)")

    if unavailable:
        click.echo("\nUnavailable (missing dependencies):")
        for module, error in sorted(unavailable.items()):
            click.echo(f"  - {module}: {error}")


@eval_group.command("run")
@click.argument("benchmark", required=False)
# Hook files
@click.option(
    "--evaluator",
    "evaluator_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to evaluator hook file (Python file exporting EvaluatorClass). Mutually exclusive with BENCHMARK.",
)
@click.option(
    "--env",
    "-e",
    "env_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to environment hook file (Python file exporting create_env_factory).",
)
# Model config
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["sglang", "bedrock", "kimi"]),
    default="sglang",
    help="Model backend.",
)
@click.option(
    "--base-url",
    type=str,
    default="http://localhost:30000",
    help="Base URL for SGLang server.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="Model ID. Auto-detected for SGLang if not provided.",
)
@click.option(
    "--tokenizer-path",
    type=str,
    default=None,
    help="Tokenizer path for SGLang. Defaults to model_id if not provided.",
)
@click.option(
    "--region",
    type=str,
    default=None,
    help="AWS region for Bedrock.",
)
@click.option(
    "--profile-name",
    type=str,
    default=None,
    help="AWS profile name for Bedrock.",
)
@click.option(
    "--role-arn",
    type=str,
    default=None,
    help="AWS role ARN for Bedrock (optional).",
)
@click.option(
    "--tool-parser",
    type=str,
    default=None,
    help="Tool parser: name (e.g., 'hermes', 'qwen_xml') or path to hook file.",
)
# Sampling params
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Sampling temperature. If not set, uses the model's default.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=16384,
    help="Maximum new tokens to generate.",
)
@click.option(
    "--top-p",
    type=float,
    default=None,
    help="Top-p sampling parameter. If not set, uses the model's default.",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-k sampling parameter.",
)
# Environment config
@click.option(
    "--system-prompt",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to system prompt file (overrides environment default).",
)
@click.option(
    "--max-tool-iters",
    type=int,
    default=None,
    help="Maximum tool iterations per step.",
)
@click.option(
    "--max-tool-calls",
    type=int,
    default=None,
    help="Maximum tool calls per step.",
)
@click.option(
    "--max-parallel-tool-calls",
    type=int,
    default=None,
    help="Maximum parallel tool calls per model response (excess are cancelled, not executed).",
)
# Eval settings
@click.option(
    "--n-samples-per-prompt",
    type=int,
    default=1,
    help="Number of samples per prompt (for pass@k).",
)
@click.option(
    "--max-concurrency",
    type=int,
    default=10,
    help="Maximum concurrent evaluations.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory. Defaults to {benchmark}_eval/.",
)
@click.option(
    "--save-interval",
    type=int,
    default=10,
    help="Save results every N samples.",
)
@click.option(
    "--keep-tokens",
    is_flag=True,
    default=False,
    help="Keep token-level observations in results.",
)
# Debug
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def run_cmd(
    benchmark: str | None,
    evaluator_path: Path | None,
    env_path: Path,
    # Model
    backend: Literal["sglang", "bedrock", "kimi"],
    base_url: str,
    model_id: str | None,
    tokenizer_path: str | None,
    region: str | None,
    profile_name: str | None,
    role_arn: str | None,
    tool_parser: str | None,
    # Sampling
    temperature: float | None,
    max_tokens: int,
    top_p: float | None,
    top_k: int | None,
    # Environment
    system_prompt: Path | None,
    max_tool_iters: int | None,
    max_tool_calls: int | None,
    max_parallel_tool_calls: int | None,
    # Eval
    n_samples_per_prompt: int,
    max_concurrency: int,
    output: Path,
    save_interval: int,
    keep_tokens: bool,
    debug: bool,
) -> None:
    """Run benchmark evaluation.

    BENCHMARK is the name of a registered benchmark (e.g., 'aime-2024').
    Alternatively, use --evaluator to specify a custom evaluator hook file.

    Examples:
        strands-env eval run aime-2024 --env my_env.py
        strands-env eval run --evaluator my_evaluator.py --env my_env.py
    """
    # Setup logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Validate: either benchmark or evaluator_path, not both, not neither
    if benchmark and evaluator_path:
        raise click.ClickException("Cannot specify both BENCHMARK and --evaluator. Use one or the other.")
    if not benchmark and not evaluator_path:
        raise click.ClickException("Must specify either BENCHMARK or --evaluator.")

    # Get evaluator class from registry or hook file
    if benchmark:
        try:
            evaluator_cls = get_benchmark(benchmark)
        except KeyError as e:
            raise click.ClickException(str(e)) from e
        benchmark_name = benchmark
    else:
        assert evaluator_path is not None
        evaluator_cls = load_evaluator_hook(evaluator_path)
        benchmark_name = evaluator_cls.benchmark_name

    # Load hook file (validate before building model factory)
    env_factory_creator = load_env_hook(env_path)

    # Build configs
    sampling_config = SamplingConfig(
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    model_config = ModelConfig(
        backend=backend,
        base_url=base_url,
        model_id=model_id,
        tokenizer_path=tokenizer_path,
        tool_parser=tool_parser,
        region=region,
        profile_name=profile_name,
        role_arn=role_arn,
        sampling=sampling_config,
    )
    env_config = EnvConfig(
        system_prompt_path=system_prompt,
        max_tool_iters=max_tool_iters,
        max_tool_calls=max_tool_calls,
        max_parallel_tool_calls=max_parallel_tool_calls,
        verbose=False,  # Always False for eval
    )
    eval_config = EvalConfig(
        n_samples_per_prompt=n_samples_per_prompt,
        max_concurrency=max_concurrency,
        output_dir=output,
        save_interval=save_interval,
        keep_tokens=keep_tokens,
    )

    # Build model factory
    model_factory = build_model_factory(model_config, eval_config.max_concurrency)

    # Create env_factory from hook
    env_factory = env_factory_creator(model_factory, env_config)

    # Get output paths based on benchmark name
    output_dir = eval_config.get_output_dir(benchmark_name)
    results_path = output_dir / "results.jsonl"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator
    evaluator = evaluator_cls(
        env_factory=env_factory,
        max_concurrency=eval_config.max_concurrency,
        n_samples_per_prompt=eval_config.n_samples_per_prompt,
        output_path=results_path,
        save_interval=eval_config.save_interval,
        keep_tokens=eval_config.keep_tokens,
    )

    # Load dataset once (convert to list since we need to iterate twice if resolving system_prompt)
    actions = list(evaluator.load_dataset())

    # Resolve system_prompt from environment if not provided via CLI
    resolved_system_prompt = env_config.system_prompt
    if resolved_system_prompt is None and actions:
        # Use first action from dataset to create environment and get system_prompt
        async def get_env_system_prompt() -> str | None:
            env = await env_factory(actions[0])
            return env.system_prompt

        resolved_system_prompt = asyncio.run(get_env_system_prompt())

    # Save config for reproducibility
    config_data = {
        "benchmark": benchmark_name,
        "evaluator_path": str(evaluator_path) if evaluator_path else None,
        "env_path": str(env_path),
        "model": model_config.to_dict(),
        "env": {
            "system_prompt": resolved_system_prompt,
            "max_tool_iters": env_config.max_tool_iters,
            "max_tool_calls": env_config.max_tool_calls,
            "max_parallel_tool_calls": env_config.max_parallel_tool_calls,
        },
        "eval": eval_config.to_dict(),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    click.echo(f"Saved config to {config_path}")

    # Run evaluation
    click.echo(f"Running {benchmark_name} evaluation with {env_path}")
    click.echo(f"  Backend: {backend}, Model: {model_id or '(auto-detect)'}")
    click.echo(f"  Samples per prompt: {n_samples_per_prompt}, Concurrency: {max_concurrency}")
    click.echo(f"  Output directory: {output_dir}")

    results = asyncio.run(evaluator.run(actions))
    metrics = evaluator.compute_metrics(results)

    # Save metrics to JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Saved metrics to {metrics_path}")
