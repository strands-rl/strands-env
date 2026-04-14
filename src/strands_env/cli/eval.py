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

"""Evaluation CLI commands."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import click

from strands_env.core.models import ModelConfig
from strands_env.eval import get_benchmark, list_benchmarks, list_unavailable_benchmarks
from strands_env.utils.loader import load_env_factory_hook, load_evaluator_hook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Click Types
# ---------------------------------------------------------------------------


class JsonType(click.ParamType):
    """Click parameter type that accepts inline JSON or a path to a JSON file."""

    name = "JSON"

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> dict:
        """Convert value to dict."""
        path = Path(value)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}", param, ctx)


JSON = JsonType()


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


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
# Hooks (dotted paths)
@click.option(
    "--evaluator",
    "evaluator_path",
    type=str,
    default=None,
    help="Dotted path to evaluator module (exporting EvaluatorClass). Mutually exclusive with BENCHMARK.",
)
@click.option(
    "--env",
    "-e",
    "env_hook",
    type=str,
    required=True,
    help="Dotted path to environment hook module (exporting create_env_factory).",
)
@click.option(
    "--env-config",
    type=JSON,
    default=None,
    help="Environment config as inline JSON or path to JSON file.",
)
# Model config
@click.option(
    "--backend", "-b", type=click.Choice(["sglang", "bedrock", "kimi"]), default="sglang", help="Model backend."
)
@click.option("--base-url", type=str, default="http://localhost:30000", help="Base URL for SGLang server.")
@click.option("--model-id", type=str, default=None, help="Model ID. Auto-detected for SGLang if not provided.")
@click.option("--tokenizer-path", type=str, default=None, help="Tokenizer path for SGLang.")
@click.option("--region", type=str, default=None, help="AWS region for Bedrock.")
@click.option("--profile-name", type=str, default=None, help="AWS profile name for Bedrock.")
@click.option("--role-arn", type=str, default=None, help="AWS role ARN for Bedrock.")
@click.option("--tool-parser", type=str, default=None, help="Tool parser name (e.g., 'hermes', 'qwen_xml').")
# Sampling params
@click.option("--temperature", type=float, default=None, help="Sampling temperature.")
@click.option("--max-tokens", type=int, default=16384, help="Maximum new tokens to generate.")
@click.option("--top-p", type=float, default=None, help="Top-p sampling parameter.")
@click.option("--top-k", type=int, default=None, help="Top-k sampling parameter.")
# Eval settings
@click.option("--n-samples-per-prompt", type=int, default=1, help="Number of samples per prompt (for pass@k).")
@click.option("--max-concurrency", type=int, default=10, help="Maximum concurrent evaluations.")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory.")
@click.option("--max-samples", type=int, default=None, help="Maximum dataset samples to evaluate.")
@click.option("--save-interval", type=int, default=10, help="Save results every N samples.")
@click.option("--keep-tokens", is_flag=True, default=False, help="Keep token-level observations in results.")
# Distributed
@click.option("--n-actors-per-node", type=int, default=None, help="Ray actors per node for distributed eval.")
# Debug
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
def run_cmd(
    benchmark: str | None,
    evaluator_path: str | None,
    env_hook: str,
    env_config: dict | None,
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
    # Eval
    n_samples_per_prompt: int,
    max_concurrency: int,
    max_samples: int | None,
    output: Path,
    save_interval: int,
    keep_tokens: bool,
    # Distributed
    n_actors_per_node: int | None,
    # Misc
    debug: bool,
) -> None:
    """Run benchmark evaluation.

    BENCHMARK is the name of a registered benchmark (e.g., 'aime-2024').
    Alternatively, use --evaluator to specify a custom evaluator module.

    Examples:
        strands-env eval run aime-2024 -e examples.eval.simple_math.calculator_env
        strands-env eval run --evaluator my_package.evaluator -e my_package.env_hook
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve evaluator; benchmark takes precedence over evaluator_path
    if benchmark:
        evaluator_cls = get_benchmark(benchmark)
        benchmark_name = benchmark
        if evaluator_path:
            logger.warning("Registered %s benchmark, ignoring --evaluator %s.", benchmark, evaluator_path)
    else:
        assert evaluator_path is not None
        evaluator_cls = load_evaluator_hook(evaluator_path)
        benchmark_name = evaluator_cls.benchmark_name

    # Load env factory hook (validate before building model)
    load_env_factory_hook(env_hook)

    # Build model config
    sampling_params: dict = {"max_new_tokens": max_tokens}
    if temperature is not None:
        sampling_params["temperature"] = temperature
    if top_p is not None:
        sampling_params["top_p"] = top_p
    if top_k is not None:
        sampling_params["top_k"] = top_k

    model_config = ModelConfig(
        backend=backend,
        base_url=base_url,
        model_id=model_id,
        tokenizer_path=tokenizer_path,
        tool_parser=tool_parser,
        max_connections=max_concurrency,
        region=region or "us-west-2",
        profile_name=profile_name,
        role_arn=role_arn,
        sampling_params=sampling_params,
    )

    # Build env_factory (local) or env_actor_pool (distributed)
    env_factory = None
    env_actor_pool = None
    if n_actors_per_node is not None:
        import ray

        from strands_env.utils.ray import EnvironmentActorPool

        if not ray.is_initialized():
            ray.init()
        env_actor_pool = EnvironmentActorPool(
            env_hook_path=env_hook + ".create_env_factory",
            env_hook_config={"model_config": model_config.to_dict(), **(env_config or {})},
            n_actors_per_node=n_actors_per_node,
        )
    else:
        env_factory_creator = load_env_factory_hook(env_hook)
        env_factory = env_factory_creator(model_config.to_dict(), **(env_config or {}))

    # Output paths
    output_dir = output or Path(f"{benchmark_name}_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator and load dataset
    evaluator = evaluator_cls(
        env_factory=env_factory,
        max_concurrency=max_concurrency,
        n_samples_per_prompt=n_samples_per_prompt,
        output_path=output_dir / "results.jsonl",
        save_interval=save_interval,
        keep_tokens=keep_tokens,
        env_actor_pool=env_actor_pool,
    )
    actions = list(evaluator.load_dataset())[:max_samples]

    # Save config for reproducibility
    config_data = {
        "benchmark": benchmark_name,
        "evaluator_path": evaluator_path,
        "env_hook": env_hook,
        "env_config": env_config or {},
        "model_config": model_config.to_dict(),
        "n_actors_per_node": n_actors_per_node,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    mode = f"distributed ({n_actors_per_node} actors/node)" if env_actor_pool else "local"
    click.echo(
        f"Running {benchmark_name} | {backend} | {model_id or '(auto)'} | "
        f"{len(actions)} samples | n={n_samples_per_prompt} | concurrency={max_concurrency} | {mode} | {output_dir}"
    )

    # Run and save metrics
    results = asyncio.run(evaluator.run(actions))
    metrics = evaluator.compute_metrics(results)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if env_actor_pool is not None:
        env_actor_pool.shutdown()

    click.echo(f"Done. Metrics saved to {output_dir / 'metrics.json'}")
