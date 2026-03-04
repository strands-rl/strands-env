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

"""Integration tests for WebSearchEnv.

Requires:
    - A running SGLang server (auto-skipped if unreachable)
    - SERPER_API_KEY env var for Serper provider tests
    - GOOGLE_API_KEY + GOOGLE_CSE_ID env vars for Google provider tests
"""

import os

import pytest

from strands_env.core.types import Action, TerminationReason
from strands_env.environments.web_search import ScrapeConfig, SearchConfig, WebSearchEnv

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

serper_available = pytest.mark.skipif(
    not os.getenv("SERPER_API_KEY"),
    reason="SERPER_API_KEY not set",
)

google_available = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    reason="GOOGLE_API_KEY and GOOGLE_CSE_ID not set",
)


# ---------------------------------------------------------------------------
# Serper provider
# ---------------------------------------------------------------------------


@serper_available
class TestSerperWebSearchEnv:
    async def test_search_step_completes(self, model_factory):
        """Agent can search the web and produce a response."""
        env = WebSearchEnv(model_factory=model_factory)
        try:
            result = await env.step(Action(message="What is the capital of France?"))
            assert result.termination_reason == TerminationReason.TASK_COMPLETE
            assert result.observation.final_response
        finally:
            await env.cleanup()

    async def test_search_and_scrape_step_completes(self, model_factory):
        """Agent can search and scrape pages."""
        env = WebSearchEnv(
            model_factory=model_factory,
            scrape_config=ScrapeConfig(),
        )
        try:
            result = await env.step(Action(message="What is the population of Tokyo?"))
            assert result.termination_reason == TerminationReason.TASK_COMPLETE
            assert result.observation.final_response
        finally:
            await env.cleanup()

    async def test_tool_iteration_limit(self, model_factory):
        """Environment respects max_tool_iters."""
        env = WebSearchEnv(
            model_factory=model_factory,
            max_tool_iters=1,
        )
        try:
            result = await env.step(
                Action(
                    message="Search for 10 different topics: Python, Java, Rust, Go, C++, Ruby, PHP, Swift, Kotlin, Scala. Search each one separately."
                )
            )
            assert result.termination_reason in (
                TerminationReason.MAX_TOOL_ITERATIONS_REACHED,
                TerminationReason.TASK_COMPLETE,
            )
        finally:
            await env.cleanup()

    async def test_step_metrics(self, model_factory):
        """Step produces expected metric keys."""
        env = WebSearchEnv(model_factory=model_factory)
        try:
            result = await env.step(Action(message="Who wrote the book 1984?"))
            metrics = result.observation.metrics
            assert "message_count" in metrics
            assert "model_calls" in metrics
            assert metrics["model_calls"] >= 1
        finally:
            await env.cleanup()


# ---------------------------------------------------------------------------
# Google provider
# ---------------------------------------------------------------------------


@google_available
class TestGoogleWebSearchEnv:
    async def test_search_step_completes(self, model_factory):
        """Agent can search with Google Custom Search and produce a response."""
        env = WebSearchEnv(
            model_factory=model_factory,
            search_config=SearchConfig(provider="google"),
        )
        try:
            result = await env.step(Action(message="What is the speed of light?"))
            assert result.termination_reason == TerminationReason.TASK_COMPLETE
            assert result.observation.final_response
        finally:
            await env.cleanup()
