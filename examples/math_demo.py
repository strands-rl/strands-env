from __future__ import annotations

import asyncio
import logging
from typing import Literal

import click

from strands_env.core.models import ModelConfig, build_model_factory
from strands_env.core.types import Task
from strands_env.environments.math import MathEnv, MathVerifyReward

MATH_PROBLEMS = [
    ("What is 123 * 456?", "56088"),
    ("What is the square root of 144?", "12"),
    ("What is 2^10?", "1024"),
]


async def run_demo(
    backend: Literal["sglang", "bedrock"],
    model_id: str | None,
    base_url: str,
) -> None:
    """Run math problems through the math environment."""
    # Build model factory using CLI utilities
    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=base_url,
        tool_parser="qwen_xml",  # for Qwen/Qwen3.5 models
        sampling_params={"max_new_tokens": 16384},
    )
    model_factory = build_model_factory(config)

    # Create environment with the math reward function
    env = MathEnv(
        model_factory=model_factory,
        reward_fn=MathVerifyReward(),
        verbose=False,
    )

    # Run each problem
    for question, ground_truth in MATH_PROBLEMS:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Question: {question}")
        click.echo(f"Expected: {ground_truth}")
        click.echo("-" * 60)

        task = Task(message=question, ground_truth=ground_truth)
        result = await env.rollout(task)

        click.echo(f"Termination: {result.termination_reason.value}")
        click.echo(f"Response:    {result.final_response}")
        click.echo(f"Reward:      {result.reward_result.reward if result.reward_result else None}")
        click.echo(f"Metrics:     {result.metrics}")


@click.command()
@click.option(
    "--backend",
    "-b",
    required=True,
    type=click.Choice(["sglang", "bedrock"]),
    help="Model backend.",
)
@click.option(
    "--model-id",
    default=None,
    help="Model ID. Auto-detected for SGLang if not provided.",
)
@click.option(
    "--base-url",
    default="http://localhost:30000",
    help="Base URL for SGLang server.",
)
def main(backend: str, model_id: str | None, base_url: str) -> None:
    """Run math problems through a math environment.

    \b
    python examples/math_demo.py --backend sglang
    python examples/math_demo.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
    """
    logging.basicConfig(level=logging.WARNING)

    asyncio.run(run_demo(backend, model_id, base_url))


if __name__ == "__main__":
    main()
