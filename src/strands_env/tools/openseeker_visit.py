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

"""OpenSeeker-style page visit tool with Jina Reader and LLM summarization.

Ported from `OpenSeeker <https://github.com/rui-ye/OpenSeeker>`_.  Fetches page
content via the Jina Reader API, then summarizes with an LLM to extract
structured evidence and summary relevant to a goal.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import aiohttp
import tiktoken
from strands import tool

from strands_env.core import Environment
from strands_env.core.types import Action

if TYPE_CHECKING:
    from strands_env.core.models import ModelFactory

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 50
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_MAX_TOKENS = 95_000

# Note: "feilds" typo preserved from the OpenSeeker training prompt to maintain model compatibility.
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


@functools.lru_cache(maxsize=1)
def _cl100k_encoding() -> tiktoken.Encoding:
    """Return the cl100k_base encoding (cached after first call)."""
    return tiktoken.get_encoding("cl100k_base")


def _truncate_to_tokens(text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Truncate `text` to at most `max_tokens` cl100k tokens."""
    tokens = _cl100k_encoding().encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text
    return _cl100k_encoding().decode(tokens[:max_tokens])


def _is_valid_content(content: str) -> bool:
    """Return ``True`` if `content` looks like a successful page fetch."""
    return bool(content) and not content.startswith("[visit] Failed") and content != "[visit] Empty content."


class OpenSeekerVisitToolkit:
    """Web page visit with Jina Reader and LLM summarization for Strands agents.

    Notes:
        - Fetches page content via Jina Reader API (``https://r.jina.ai/{url}``).
        - Summarizes content using an LLM via a nested `Environment` call.
        - Output format matches OpenSeeker's structured evidence + summary format.
        - A shared `aiohttp.ClientSession` and `asyncio.Semaphore` cap concurrent
          requests.  Call `cleanup` when done.
    """

    def __init__(
        self,
        *,
        summarizer_model_factory: ModelFactory | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        semaphore: asyncio.Semaphore | None = None,
        max_summary_retries: int = 3,
    ) -> None:
        """Initialize an `OpenSeekerVisitToolkit` instance.

        Args:
            summarizer_model_factory: Factory for creating model instances used
                by the nested LLM summarizer.  When ``None``, raw page text
                (truncated to 5 000 tokens) is returned instead.
            timeout: HTTP request timeout in seconds for Jina Reader calls.
            max_concurrency: Max concurrent requests (ignored if *semaphore*
                is provided).
            semaphore: Shared semaphore for global rate limiting across toolkit
                instances.
            max_summary_retries: Max retries with progressive truncation when
                the LLM returns a too-short summary.
        """
        self._summarizer_model_factory = summarizer_model_factory
        self._timeout = timeout
        self._semaphore = semaphore or asyncio.Semaphore(max_concurrency)
        self._session: aiohttp.ClientSession | None = None
        self._max_summary_retries = max_summary_retries

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

    # ------------------------------------------------------------------
    # Jina page fetching
    # ------------------------------------------------------------------

    async def _jina_fetch_once(self, url: str) -> str:
        """Single Jina Reader API attempt with inner retries for transient errors."""
        jina_api_key = os.environ.get("JINA_API_KEY", "")
        headers = {"Authorization": f"Bearer {jina_api_key}"} if jina_api_key else {}

        inner_retries = 3
        for attempt in range(inner_retries):
            try:
                async with self._semaphore:
                    async with self._get_session().get(f"https://r.jina.ai/{url}", headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.text()
                        raise ValueError(f"Jina returned status {resp.status}")
            except Exception as e:
                logger.warning(
                    "[jina_fetch] inner attempt %d/%d for %s: %s",
                    attempt + 1,
                    inner_retries,
                    url,
                    e,
                )
                if attempt == inner_retries - 1:
                    return "[visit] Failed to read page."
                await asyncio.sleep(0.5)

        return "[visit] Failed to read page."

    async def _jina_fetch(self, url: str) -> str:
        """Fetch page content via Jina Reader API with outer retry loop.

        Matches OpenSeeker's ``html_readpage_jina``: up to 8 outer attempts,
        each with 3 inner retries on the Jina API call.
        """
        max_outer_attempts = 8
        for _attempt in range(max_outer_attempts):
            content = await self._jina_fetch_once(url)
            if _is_valid_content(content):
                return content
        return "[visit] Failed to read page."

    # ------------------------------------------------------------------
    # LLM summarization
    # ------------------------------------------------------------------

    async def _summarize(self, content: str, goal: str) -> str:
        """Summarize page content using an LLM via a nested `Environment`."""
        if self._summarizer_model_factory is None:
            logger.warning("`summarizer_model_factory` is not set. Returning raw content (truncated).")
            return _truncate_to_tokens(content, max_tokens=5000)

        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        environment = Environment(model_factory=self._summarizer_model_factory)
        result = await environment.step(action=Action(message=prompt))
        return result.observation.final_response or ""

    # ------------------------------------------------------------------
    # Single URL visit
    # ------------------------------------------------------------------

    async def _visit_single(self, url: str, goal: str) -> str:
        """Fetch a single URL and return structured evidence + summary.

        Matches OpenSeeker's ``readpage_jina`` retry logic:

        1. Summarize with progressive truncation (up to 3 retries if summary
           is too short).
        2. Parse JSON with up to 3 additional LLM retries if JSON parsing fails.
        """
        raw_content = await self._jina_fetch(url)

        if not _is_valid_content(raw_content):
            logger.warning("[visit] Failed to visit: %s", url)
            return f"[visit] Failed to visit the url: {url}"

        content = _truncate_to_tokens(raw_content, max_tokens=DEFAULT_MAX_TOKENS)

        # Phase 1: Summarize with progressive truncation on short responses
        raw_summary = await self._summarize(content, goal)
        summary_retries = self._max_summary_retries
        while len(raw_summary) < 10 and summary_retries > 0:
            truncate_length = int(0.7 * len(content)) if summary_retries > 1 else 25_000
            logger.info(
                "[visit] Summary retry for %s, attempt %d/%d, truncating to %d chars",
                url,
                self._max_summary_retries - summary_retries + 1,
                self._max_summary_retries,
                truncate_length,
            )
            content = content[:truncate_length]
            raw_summary = await self._summarize(content, goal)
            summary_retries -= 1

        # Phase 2: Parse JSON, retry with additional LLM calls if parsing fails
        parsed = self._parse_summary(raw_summary)
        parse_retries = 3
        while parsed is None and parse_retries > 0:
            logger.info("[visit] JSON parse retry for %s, %d retries left", url, parse_retries)
            raw_summary = await self._summarize(content, goal)
            parsed = self._parse_summary(raw_summary)
            parse_retries -= 1

        if parsed is None:
            return (
                f"The useful information in {url} for user goal {goal} as follows: \n\n"
                "Evidence in page: \nThe provided webpage content could not be accessed. "
                "Please check the URL or file format.\n\n"
                "Summary: \nThe webpage content could not be processed, "
                "and therefore, no information is available.\n\n"
            )

        return (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{parsed.get('evidence', '')}\n\n"
            f"Summary: \n{parsed.get('summary', '')}\n\n"
        )

    @staticmethod
    def _parse_summary(raw: str) -> dict[str, Any] | None:
        """Try to parse a JSON summary from the LLM response."""
        if not raw:
            return None

        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            # Try extracting JSON from surrounding text.
            left = cleaned.find("{")
            right = cleaned.rfind("}")
            if left != -1 and right != -1 and left <= right:
                try:
                    return json.loads(cleaned[left : right + 1])  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    pass
        return None

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    @tool
    async def visit(self, url: str | list[str], goal: str) -> str:
        """Parse webpage(s) and return the summary of the content according to the goal.

        Args:
            url: The URL(s) of the webpage(s) to visit. Can be a single URL
                or an array of URLs.
            goal: The goal of the visit for webpage(s).

        Returns:
            Structured evidence and summary extracted from the page(s).
        """
        logger.info("[openseeker_visit] url=%s, goal=%s", url, goal[:100] if goal else "")

        if isinstance(url, str):
            return await self._visit_single(url, goal)

        # Batch visit with 900 s total timeout (matching OpenSeeker).
        results: list[str] = []
        start_time = time.monotonic()
        for u in url:
            if time.monotonic() - start_time > 900:
                results.append(
                    f"The useful information in {u} for user goal {goal} as follows: \n\n"
                    "Evidence in page: \nThe provided webpage content could not be accessed. "
                    "Please check the URL or file format.\n\n"
                    "Summary: \nThe webpage content could not be processed, "
                    "and therefore, no information is available.\n\n"
                )
            else:
                try:
                    result = await self._visit_single(u, goal)
                except Exception as e:
                    result = f"Error fetching {u}: {e}"
                results.append(result)

        return "\n---\n".join(results)
