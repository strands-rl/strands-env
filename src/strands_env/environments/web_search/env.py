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

"""Web search environment with web search and web scraping tools."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import override

from strands_env.core.environment import Environment
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.tools.web_scraper import WebScraperToolkit
from strands_env.tools.web_search import WebSearchToolkit


@dataclass
class SearchConfig:
    """Configuration for the web search tool provider."""

    timeout: int = 10
    max_concurrency: int = 10
    semaphore: asyncio.Semaphore | None = None
    blocked_domains: list[str] | None = None
    provider: Literal["serper", "google"] = "serper"

    def _search_tool_name(self) -> str:
        return f"{self.provider}_search"


@dataclass
class ScrapeConfig:
    """Configuration for the web scraping tool."""

    timeout: int = 30
    max_concurrency: int = 10
    semaphore: asyncio.Semaphore | None = None
    token_budget: int = 5000
    summarizer_model_factory: ModelFactory | None = None

    def _scrape_tool_name(self) -> str:
        return "scrape" if self.summarizer_model_factory is None else "scrape_and_summarize"


class WebSearchEnv(Environment):
    """Web search environment with pluggable search providers."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = 5,
        max_tool_calls: int | None = 10,
        verbose: bool = False,
        search_config: SearchConfig | None = None,
        scrape_config: ScrapeConfig | None = None,
    ):
        """Initialize the web search environment with search and optional scrape configurations."""
        if search_config is None:
            search_config = SearchConfig()
        super().__init__(
            model_factory=model_factory,
            system_prompt=system_prompt,
            reward_fn=reward_fn,
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )
        # By default, only use the search tool.
        self._search_tool_name = search_config._search_tool_name()
        self.search_toolkit = WebSearchToolkit(
            timeout=search_config.timeout,
            max_concurrency=search_config.max_concurrency,
            semaphore=search_config.semaphore,
            blocked_domains=search_config.blocked_domains,
        )
        # If scrape_config is provided, use the scrape tool.
        self._scrape_tool_name: str | None = None
        self.scraper_toolkit: WebScraperToolkit | None = None
        if scrape_config is not None:
            self._scrape_tool_name = scrape_config._scrape_tool_name()
            self.scraper_toolkit = WebScraperToolkit(
                token_budget=scrape_config.token_budget,
                timeout=scrape_config.timeout,
                max_concurrency=scrape_config.max_concurrency,
                semaphore=scrape_config.semaphore,
                summarizer_model_factory=scrape_config.summarizer_model_factory,
            )

    @override
    def get_tools(self) -> list:
        tools = [getattr(self.search_toolkit, self._search_tool_name)]
        if self.scraper_toolkit is not None:
            assert self._scrape_tool_name is not None
            tools.append(getattr(self.scraper_toolkit, self._scrape_tool_name))
        return tools

    async def cleanup(self) -> None:
        """Close shared HTTP sessions for all toolkits."""
        await self.search_toolkit.cleanup()
        if self.scraper_toolkit is not None:
            await self.scraper_toolkit.cleanup()
