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

"""Web search toolkit with different search providers."""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from strands import tool

from strands_env.utils.decorators import requires_env

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
DEFAULT_MAX_CONCURRENCY = 10
MAX_RESULTS = 10

GOOGLE_SERPER_DEV_URL = "https://google.serper.dev/search"
GOOGLE_CUSTOM_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


class WebSearchToolkit:
    """Web search tools supporting different search providers.

    Supports multiple search providers.  Each provider is exposed as a
    separate `@tool` method so the environment can pick which one to
    use.  Credentials are validated lazily — only when the corresponding
    tool method is actually called.

    A single shared `aiohttp.ClientSession` (created lazily) and
    an `asyncio.Semaphore` cap concurrent requests.  Call
    `cleanup` when done to close the session.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        semaphore: asyncio.Semaphore | None = None,
        blocked_domains: list[str] | None = None,
    ):
        """Initialize a `WebSearchToolkit` instance.

        All credential parameters are optional.  When `None`, the
        corresponding environment variable is checked.  Validation only
        happens when a tool method that needs the credential is called.

        Args:
            timeout: HTTP request timeout in seconds.
            max_concurrency: Max concurrent requests (ignored if *semaphore* is provided).
            semaphore: Shared semaphore for global rate limiting across toolkit instances.
            blocked_domains: Domains to exclude from results (e.g. `["huggingface.co"]`).
        """
        self._timeout = timeout
        self._semaphore = semaphore or asyncio.Semaphore(max_concurrency)
        self._blocked_domains = blocked_domains or []
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def cleanup(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _apply_blocked_domains(self, query: str) -> str:
        """Append `-site:` exclusions to *query* for blocked domains."""
        if self._blocked_domains:
            return query + " " + " ".join(f"-site:{d}" for d in self._blocked_domains)
        return query

    @staticmethod
    def format_results(
        items: list[dict], *, title_key: str = "title", url_key: str = "link", snippet_key: str = "snippet"
    ) -> str:
        """Format a list of search result dicts into a numbered text block.

        This is a hook method that can be overridden by subclasses to customize the formatting.
        """
        if not items:
            return "No results found."
        lines = []
        for i, item in enumerate(items, 1):
            title = item.get(title_key) or "No title available."
            url = item.get(url_key) or "No URL available."
            snippet = item.get(snippet_key) or "No snippet available."
            lines.append(f"{i}. {title} ({url}):\n{snippet}")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Serper
    # ------------------------------------------------------------------

    @tool
    @requires_env("SERPER_API_KEY")
    async def serper_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using Serper.dev API.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info("[serper_search] query=%s, top_k=%s", query, top_k)

        query = self._apply_blocked_domains(query)

        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": top_k}

        try:
            async with self._semaphore:
                async with self._get_session().post(GOOGLE_SERPER_DEV_URL, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

            return self.format_results(data.get("organic", []))
        except Exception as e:
            logger.error("[serper_search] error: %s", e)
            return f"Search failed: {e}."

    # ------------------------------------------------------------------
    # Google Custom Search
    # ------------------------------------------------------------------

    @tool
    @requires_env("GOOGLE_API_KEY", "GOOGLE_CSE_ID")
    async def google_search(self, query: str, top_k: int = 5) -> str:
        """Search the web using Google Custom Search JSON API.

        Args:
            query: The search query.
            top_k: Number of results to return (max 10).

        Returns:
            Search results with title, URL, and snippet for each result.
        """
        logger.info("[google_search] query=%s, top_k=%s", query, top_k)

        top_k = min(top_k, MAX_RESULTS)
        query = self._apply_blocked_domains(query)

        params = {
            "key": os.environ["GOOGLE_API_KEY"],
            "cx": os.environ["GOOGLE_CSE_ID"],
            "q": query,
            "num": str(top_k),
        }

        try:
            async with self._semaphore:
                async with self._get_session().get(GOOGLE_CUSTOM_SEARCH_URL, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            return self.format_results(data.get("items", []))
        except Exception as e:
            logger.error("[google_search] error: %s", e)
            return f"Search failed: {e}."
