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

"""Web search environment with web search and web scraping tools."""

import asyncio
from pathlib import Path
from typing import Literal

from typing_extensions import Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.tools.web_scraper import WebScraperToolkit
from strands_env.tools.web_search import WebSearchToolkit


class WebSearchConfig(EnvironmentConfig, total=False):
    """Serializable configuration for `WebSearchEnv`."""

    # Search
    search_provider: Literal["serper", "google"]
    search_timeout: int
    blocked_domains: list[str]

    # Scrape
    scrape_enabled: bool
    scrape_timeout: int
    scrape_token_budget: int


class WebSearchEnv(Environment):
    """Web search environment with pluggable search providers."""

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
        """Initialize a `WebSearchEnv` instance."""
        super().__init__(model_factory=model_factory, reward_fn=reward_fn, **config)  # type: ignore[misc]

        provider: str = self.config.get("search_provider", "serper")
        self.search_toolkit = WebSearchToolkit(
            timeout=int(self.config.get("search_timeout", 10)),
            concurrency=search_concurrency,
            blocked_domains=self.config.get("blocked_domains"),  # type: ignore[arg-type]
        )
        self.search_tool = getattr(self.search_toolkit, f"{provider}_search")

        self.scrape_tool = None
        self.scraper_toolkit: WebScraperToolkit | None = None
        if self.config.get("scrape_enabled", False):
            self.scraper_toolkit = WebScraperToolkit(
                timeout=int(self.config.get("scrape_timeout", 30)),
                concurrency=scrape_concurrency,
                token_budget=int(self.config.get("scrape_token_budget", 5000)),
                summarizer_model_factory=summarizer_model_factory,
            )
            self.scrape_tool = (
                self.scraper_toolkit.scrape_and_summarize if summarizer_model_factory else self.scraper_toolkit.scrape
            )

    @override
    def get_tools(self) -> list:
        """Return search and optionally scrape tools."""
        tools = [self.search_tool]
        if self.scrape_tool is not None:
            tools.append(self.scrape_tool)
        return tools

    async def cleanup(self) -> None:
        """Close shared HTTP sessions for all toolkits."""
        await self.search_toolkit.cleanup()
        if self.scraper_toolkit is not None:
            await self.scraper_toolkit.cleanup()
