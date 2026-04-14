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

"""Calculator environment demo — shows how to use an Environment programmatically.

This example demonstrates:
- Creating a model factory
- Instantiating an environment with tools
- Running steps and inspecting results

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/calculator_demo.py --backend sglang

    # Bedrock backend (requires AWS credentials)
    python examples/calculator_demo.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import click

from strands_env.core.models import ModelConfig, build_model_factory
from strands_env.core.types import Action, TaskContext
from strands_env.environments.calculator.env import CalculatorEnv
from strands_env.rewards import MathVerifyReward

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
    """Run math problems through the calculator environment."""
    # Build model factory using CLI utilities
    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=base_url,
        tool_parser="qwen_xml",  # for Qwen/Qwen3.5 models
        sampling_params={"max_new_tokens": 16384},
    )
    model_factory = build_model_factory(config)

    # Create environment with calculator tool and math reward function
    env = CalculatorEnv(
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

        action = Action(message=question, task_context=TaskContext(ground_truth=ground_truth))
        result = await env.step(action)

        click.echo(f"Termination: {result.termination_reason.value}")
        click.echo(f"Response:    {result.observation.final_response}")
        click.echo(f"Reward:      {result.reward.reward if result.reward else None}")
        click.echo(f"Metrics:     {result.observation.metrics}")


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
    """Run math problems through a calculator environment."""
    logging.basicConfig(level=logging.WARNING)

    asyncio.run(run_demo(backend, model_id, base_url))


if __name__ == "__main__":
    main()
