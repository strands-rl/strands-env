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

"""Web scraper toolkit with optional LLM-based content extraction.

Fetches a web page, extracts main content (stripping nav/sidebar/ads),
and optionally uses a strands Agent to extract task-relevant information.

Content extraction pipeline:
  1. trafilatura: extracts main content, strips boilerplate (primary)
  2. html2text: full HTML-to-Markdown conversion (fallback)

Example:
    >>> from strands_env.tools import WebScraperToolkit
    >>> toolkit = WebScraperToolkit()
    >>> result = toolkit.scrape("https://example.com")
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiohttp
import tiktoken
from pydantic import BaseModel, Field
from strands import Agent, tool

if TYPE_CHECKING:
    from strands_env.core.models import ModelFactory

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_TOKEN_BUDGET = 5000

EXTRACT_PROMPT_TEMPLATE = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
"""


class WebPageSummary(BaseModel):
    """Structured page extraction — rationale, supporting evidence, and concise summary."""

    rationale: str = Field(description="Specific sections/data directly related to the user's goal.")
    evidence: str = Field(
        description="Most relevant information from the page, preserving full original context where possible."
    )
    summary: str = Field(description="Concise paragraph with logical flow, judging the contribution to the goal.")


class WebScraperToolkit:
    """Web scraper with optional LLM extraction for strands agents.

    Notes:
        - Two `@tool` methods are provided — the environment picks which to expose:
          `scrape` (fetch + extract) and `scrape_and_summarize` (fetch + extract + LLM,
          requires `summarizer_model_factory`).
        - A single shared `aiohttp.ClientSession` (created lazily) and an
          `asyncio.Semaphore` cap concurrent requests. Call `cleanup` when done.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        concurrency: asyncio.Semaphore | int = DEFAULT_MAX_CONCURRENCY,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        summarizer_model_factory: ModelFactory | None = None,
    ):
        """Initialize a `WebScraperToolkit` instance.

        Args:
            timeout: HTTP request timeout in seconds.
            concurrency: Semaphore or max concurrent requests for rate limiting.
            token_budget: Max tokens of page content to keep after extraction.
            summarizer_model_factory: Optional factory for creating model instances for LLM summarization.
        """
        self.timeout = timeout
        self.semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)
        self.token_budget = token_budget
        self.summarizer_model_factory = summarizer_model_factory
        self._session: aiohttp.ClientSession | None = None
        self._encoding = tiktoken.encoding_for_model("gpt-4")

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self._session

    async def cleanup(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def fetch_html(self, url: str) -> str:
        """Fetch a web page and return the HTML."""
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        async with self.semaphore:
            async with self._get_session().get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()

    async def extract_content(self, html: str, url: str) -> str:
        """Extract main content from HTML, stripping boilerplate and truncating to token budget.

        Notes:
            - Uses `trafilatura` as primary extractor; falls back to `html2text`
              for pages where `trafilatura` returns insufficient content.
            - A fresh `html2text` instance is created per call for thread safety
              (runs in a thread pool via `asyncio.to_thread`).
        """
        import html2text  # type: ignore[import-untyped]
        import trafilatura  # type: ignore[import-untyped]

        def _truncate(text: str) -> str:
            tokens = self._encoding.encode(text)
            if len(tokens) > self.token_budget:
                return self._encoding.decode(tokens[: self.token_budget]) + "...(content truncated)"
            return text

        content = await asyncio.to_thread(
            trafilatura.extract,
            html,
            url=url,
            include_links=True,
            include_tables=True,
            output_format="txt",
        )
        if content and len(content.strip()) > 100:
            return _truncate(content)

        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = True
        h2t.ignore_emphasis = False
        h2t.body_width = 0
        content = await asyncio.to_thread(h2t.handle, html)
        return _truncate(content)

    async def summarize(self, content: str, goal: str) -> WebPageSummary | None:
        """Extract structured evidence + summary for a goal using a LLM."""
        if self.summarizer_model_factory is None:
            raise RuntimeError("`summarizer_model_factory` is required for summarization.")

        prompt = EXTRACT_PROMPT_TEMPLATE.format(content=content, goal=goal)
        summarizer = Agent(model=self.summarizer_model_factory(), tools=[])
        try:
            return await summarizer.structured_output_async(output_model=WebPageSummary, prompt=prompt)
        except Exception as e:
            logger.error("[web_page_summary] error: content=%s..., goal=%s..., error=%s", content[:100], goal[:100], e)
            return None

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    async def scrape(self, url: str) -> str:
        """Fetch a web page and extract its main content.

        Retrieves the full HTML, strips boilerplate and returns
        the extracted content.

        Args:
            url: The URL of the web page to scrape.

        Returns:
            Extracted page content or an error message.
        """
        logger.info("[scrape] url=%s", url)

        try:
            html = await self.fetch_html(url)
            content = await self.extract_content(html, url)
            return content
        except Exception as e:
            logger.error("[scrape] error: url=%s, error=%s", url, e)
            return f"Scrape failed for {url}: {e}"

    @tool
    async def scrape_and_summarize(self, url: str, instruction: str) -> str:
        """Fetch a web page, extract content, and summarize with an LLM.

        Retrieves the full HTML, strips boilerplate, then uses an LLM agent
        to extract only the information relevant to the instruction.

        Args:
            url: The URL of the web page to scrape.
            instruction: What information to extract from the page.

        Returns:
            LLM-summarized content or an error message.
        """
        logger.info("[scrape_and_summarize] url=%s, instruction=%s", url, instruction[:100])

        try:
            html = await self.fetch_html(url)
            main_content = await self.extract_content(html, url)
            content = await self.summarize(main_content, instruction)
            return content
        except Exception as e:
            logger.error("[scrape_and_summarize] error: url=%s, error=%s", url, e)
            return f"Scrape failed for {url}: {e}"
