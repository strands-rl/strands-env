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

"""Web search environment demo — search the web and scrape pages for answers.

This example demonstrates:
- Creating a WebSearchEnv with search + scrape tools
- Running steps and inspecting results
- Cleaning up HTTP sessions

Requires SERPER_API_KEY (or GOOGLE_API_KEY + GOOGLE_CSE_ID) env var.

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/web_search_demo.py --backend sglang

    # Bedrock backend (requires AWS credentials)
    python examples/web_search_demo.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

import click
from dotenv import load_dotenv

from strands_env.core.models import ModelConfig, build_model_factory
from strands_env.core.types import Action
from strands_env.environments.web_search import WebSearchEnv

QUESTION = "What are the key features announced in the latest Python 3.13 release?"

SYSTEM_PROMPT = (
    "You are a research assistant. First search the web, then ALWAYS use the scrape tool "
    "to read the most relevant page before answering. Cite your source URL."
)


async def run_demo(
    backend: Literal["sglang", "bedrock"],
    model_id: str | None,
    base_url: str,
) -> None:
    """Run questions through the web search environment."""
    config = ModelConfig(
        backend=backend,
        model_id=model_id,
        base_url=base_url,
        sampling_params={"max_new_tokens": 16384},
    )
    model_factory = build_model_factory(config)

    # Create environment with search + scrape tools
    env = WebSearchEnv(
        model_factory=model_factory,
        system_prompt=SYSTEM_PROMPT,
        scrape_enabled=True,
        verbose=False,
    )

    try:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Question: {QUESTION}")
        click.echo("-" * 60)

        result = await env.step(Action(message=QUESTION))

        click.echo(f"\nMessages:    {result.observation.messages}")
        click.echo(f"\nTermination: {result.termination_reason.value}")
        click.echo(f"Metrics:     {result.observation.metrics}")
    finally:
        await env.cleanup()


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
    """Run questions through a web search environment with search + scrape tools."""
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)

    asyncio.run(run_demo(backend, model_id, base_url))


if __name__ == "__main__":
    main()
