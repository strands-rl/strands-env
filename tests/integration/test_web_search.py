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

"""Integration tests for WebSearchEnv.

Requires:
    - A running SGLang server (auto-skipped if unreachable)
    - SERPER_API_KEY env var for Serper provider tests
    - GOOGLE_API_KEY + GOOGLE_CSE_ID env vars for Google provider tests
"""

import os

import pytest

from strands_env.core.types import Action, TerminationReason
from strands_env.environments.web_search import WebSearchEnv

from .conftest import assert_successful_step

serper_available = pytest.mark.skipif(not os.getenv("SERPER_API_KEY"), reason="SERPER_API_KEY not set")
google_available = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID")),
    reason="GOOGLE_API_KEY and GOOGLE_CSE_ID not set",
)


@serper_available
class TestSerperWebSearchEnv:
    async def test_search_completes_with_response(self, model_factory):
        """Agent searches the web, produces a response, and reports metrics."""
        env = WebSearchEnv(model_factory=model_factory)
        try:
            result = await env.step(Action(message="What is the capital of France?"))

            assert_successful_step(result)
            assert result.observation.metrics["model_calls"] >= 1
        finally:
            await env.cleanup()

    async def test_search_and_scrape(self, model_factory):
        """Agent can search and scrape pages when scrape_config is provided."""
        env = WebSearchEnv(model_factory=model_factory, scrape_enabled=True)
        try:
            result = await env.step(Action(message="What is the population of Tokyo?"))
            assert_successful_step(result)
        finally:
            await env.cleanup()

    async def test_tool_iteration_limit(self, model_factory):
        """max_tool_iters constrains the search agent."""
        env = WebSearchEnv(model_factory=model_factory, max_tool_iters=1)
        try:
            result = await env.step(
                Action(
                    message=(
                        "Search for 10 different topics: Python, Java, Rust, Go, C++, "
                        "Ruby, PHP, Swift, Kotlin, Scala. Search each one separately."
                    )
                ),
            )
            assert result.termination_reason in (
                TerminationReason.MAX_TOOL_ITERATIONS_REACHED,
                TerminationReason.TASK_COMPLETE,
            )
        finally:
            await env.cleanup()


@google_available
class TestGoogleWebSearchEnv:
    async def test_search_completes_with_response(self, model_factory):
        """Agent can search with Google Custom Search and produce a response."""
        env = WebSearchEnv(model_factory=model_factory, search_provider="google")
        try:
            result = await env.step(Action(message="What is the speed of light?"))
            assert_successful_step(result)
        finally:
            await env.cleanup()
