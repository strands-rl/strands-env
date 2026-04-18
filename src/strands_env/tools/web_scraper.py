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

"""Web scraper toolkit with LLM structured summarization."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import aiohttp
import tiktoken
from pydantic import BaseModel, Field
from strands import Agent, tool

from strands_env.utils.decorators import requires_env

if TYPE_CHECKING:
    from strands_env.core.models import ModelFactory

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 50
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_TOKEN_BUDGET = 20000
JINA_READER_URL = "https://r.jina.ai/{url}"

# Template for the result of the scrape tool
RESULT_TEMPLATE = """The useful information in {url} for user goal {goal} as follows:

Evidence in page:
{evidence}

Summary:
{summary}
"""

# Template for the prompt of the summarizer model
SUMMARY_PROMPT_TEMPLATE = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.
"""

TOKEN_ENCODING = tiktoken.encoding_for_model("gpt-4")


class WebPageSummary(BaseModel):
    """Structured webpage summary — rationale, supporting evidence, and concise summary."""

    rationale: str = Field(description="Specific sections/data directly related to the user's goal.")
    evidence: str = Field(
        description="Most relevant information from the page, preserving full original context where possible."
    )
    summary: str = Field(description="Concise paragraph with logical flow, judging the contribution to the goal.")


class WebScraperToolkit:
    """Web scraper with LLM-based structured summarization.

    Notes:
        - When a `summarizer_model_factory` is set, runs an LLM to produce structured
          output for the supplied goal based on fetched webpage content.
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

    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within a token budget."""
        tokens = TOKEN_ENCODING.encode(text, allowed_special="all")
        if len(tokens) > self.token_budget:
            return TOKEN_ENCODING.decode(tokens[: self.token_budget]) + "..."
        return text

    async def fetch_html(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        max_retries: int = 8,
        retry_delay: float = 0.5,
    ) -> str:
        """Fetch a URL and return the response text, retrying on transient errors.

        Retries on exceptions and empty response bodies.

        Args:
            url: The URL to fetch. Callers may pass a provider-wrapped URL (e.g. `https://r.jina.ai/{target}`) directly.
            headers: Request headers. If `None`, only aiohttp defaults are sent.
            max_retries: Total attempts before giving up.
            retry_delay: Seconds to sleep between failed attempts.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    async with self._get_session().get(url, headers=headers) as response:
                        response.raise_for_status()
                        text = await response.text()
                if not text.strip():
                    raise ValueError("empty response body")
                return text
            except Exception as e:
                last_exc = e
                logger.warning("[fetch_html] attempt %d/%d for %s: %s", attempt + 1, max_retries, url, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
        assert last_exc is not None
        raise last_exc

    async def summarize(self, content: str, goal: str) -> WebPageSummary | None:
        """Extract structured evidence + summary for a goal using a LLM."""
        if self.summarizer_model_factory is None:
            raise RuntimeError("`summarizer_model_factory` is required for summarization.")

        prompt = SUMMARY_PROMPT_TEMPLATE.format(content=content, goal=goal)
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
    @requires_env("JINA_API_KEY")
    async def scrape(self, url: str | list[str], goal: str) -> str:
        """Fetch webpage(s) and return the summary of the content.

        Args:
            url: The URL(s) of the webpage(s) to visit. Single URL or list.
            goal: What to learn from the page(s).
        """
        null_evidence = "The provided webpage content could not be accessed. Please check the URL or file format."
        null_summary = "The webpage content could not be processed, and therefore, no information is available."
        headers = {"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"}

        async def _scrape_one(u: str) -> str:
            logger.info("[scrape] url=%s, goal=%s", u, goal[:100] if goal else "")

            # Step 1: Fetch the HTML content
            try:
                raw = await self.fetch_html(JINA_READER_URL.format(url=u), headers=headers)
            except Exception as e:
                logger.error("[scrape] fetch error: url=%s, error=%s", u, e)
                return RESULT_TEMPLATE.format(url=u, goal=goal, evidence=null_evidence, summary=null_summary)

            # Step 2: Truncate the content to fit within the token budget
            content = self.truncate_text(raw)

            # Step 3: Summarize the content using the LLM
            if self.summarizer_model_factory is None:
                logger.warning("`summarizer_model_factory` is not set; returning raw content.")
                return content

            summary = await self.summarize(content, goal)
            if summary is None:
                return RESULT_TEMPLATE.format(url=u, goal=goal, evidence=null_evidence, summary=null_summary)
            return RESULT_TEMPLATE.format(url=u, goal=goal, evidence=summary.evidence, summary=summary.summary)

        if isinstance(url, list):
            results = await asyncio.gather(*(_scrape_one(u) for u in url))
            return "\n---\n".join(results)
        return await _scrape_one(url)
