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

"""Web-search environment: search and optional scrape tools behind pluggable providers."""

import asyncio
from pathlib import Path
from typing import Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction

from .tools import WebScraperToolkit, WebSearchAPIProvider, WebSearchToolkit


class WebSearchConfig(EnvironmentConfig, total=False):
    """Serializable configuration for `WebSearchEnv`."""

    # Search
    search_provider: WebSearchAPIProvider  # default "serper"
    search_timeout: int  # seconds per search API call (default 10)
    blocked_domains: list[str]  # appended to queries as -site: exclusions

    # Scrape
    scrape_enabled: bool  # default False — no scrape tool unless enabled
    scrape_timeout: int  # seconds per page fetch (default 50)
    scrape_token_budget: int  # max page tokens kept for summarization (default 20000)


class WebSearchEnv(Environment):
    """Web-search environment with a `search` tool and an optional `scrape` tool."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        search_concurrency: asyncio.Semaphore | int = 10,
        scrape_concurrency: asyncio.Semaphore | int = 10,
        summarizer_model_factory: ModelFactory | None = None,
        **config: Unpack[WebSearchConfig],
    ):
        """Initialize a `WebSearchEnv` instance.

        Args:
            model_factory: Factory for the agent's model.
            reward_fn: Optional reward function (None = inference-only).
            search_concurrency: Cap on concurrent search API calls. Pass a shared
                `asyncio.Semaphore` to enforce one budget across many envs; an int
                creates a per-env semaphore.
            scrape_concurrency: Same as `search_concurrency`, for page fetches.
            summarizer_model_factory: Model factory for the scrape tool's structured
                summarization; without it the scrape tool returns raw page content.
            **config: See `WebSearchConfig`.
        """
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)  # type: ignore[misc]

        self.search_toolkit = WebSearchToolkit(
            timeout=int(self.config.get("search_timeout", 10)),
            concurrency=search_concurrency,
            blocked_domains=self.config.get("blocked_domains"),  # type: ignore[arg-type]
            api_provider=self.config.get("search_provider", "serper"),
        )
        self.search_tool = self.search_toolkit.search

        self.scrape_tool = None
        self.scraper_toolkit: WebScraperToolkit | None = None
        if self.config.get("scrape_enabled", False):
            self.scraper_toolkit = WebScraperToolkit(
                timeout=int(self.config.get("scrape_timeout", 50)),
                concurrency=scrape_concurrency,
                token_budget=int(self.config.get("scrape_token_budget", 20000)),
                summarizer_model_factory=summarizer_model_factory,
            )
            self.scrape_tool = self.scraper_toolkit.scrape

    @override
    def get_tools(self) -> list:
        """Return the search tool, plus scrape when enabled."""
        if self.scrape_tool is not None:
            return [self.search_tool, self.scrape_tool]
        return [self.search_tool]

    async def cleanup(self) -> None:
        """Close the toolkits' shared HTTP sessions — once, after all rollouts."""
        await self.search_toolkit.cleanup()
        if self.scraper_toolkit is not None:
            await self.scraper_toolkit.cleanup()
