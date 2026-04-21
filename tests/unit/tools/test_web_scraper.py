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

"""Unit tests for `WebScraperToolkit`.

Some tests compare our behavior to OpenSeeker's upstream `visit.py` to flag
unintended divergences. Upstream reference:
https://github.com/rui-ye/OpenSeeker/blob/main/src/tools/visit.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken

from strands_env.tools.web_scraper import (
    RESULT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    TOKEN_ENCODING,
    WebPageSummary,
    WebScraperToolkit,
)


def _openseeker_truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Reference implementation copied verbatim from OpenSeeker's `visit.py`."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


class TestTruncateText:
    """`truncate_text` parity against OpenSeeker's `truncate_to_tokens`."""

    def test_short_text_unchanged(self):
        toolkit = WebScraperToolkit(token_budget=100)
        assert toolkit.truncate_text("hello world") == "hello world"

    def test_short_text_matches_upstream(self):
        toolkit = WebScraperToolkit(token_budget=100)
        text = "the quick brown fox"
        assert toolkit.truncate_text(text) == _openseeker_truncate_to_tokens(text, 100)

    def test_long_text_truncates_to_budget(self):
        toolkit = WebScraperToolkit(token_budget=10)
        text = "word " * 5000
        result = toolkit.truncate_text(text)
        result_without_suffix = result.removesuffix("...")
        token_count = len(TOKEN_ENCODING.encode(result_without_suffix, allowed_special="all"))
        assert token_count <= 10

    def test_truncation_appends_suffix_divergence(self):
        """We diverge intentionally from upstream by appending '...' on truncation.

        Upstream returns decoded tokens with no suffix. This test locks in the
        divergence so a future revert is intentional.
        """
        toolkit = WebScraperToolkit(token_budget=10)
        text = "word " * 5000
        ours = toolkit.truncate_text(text)
        upstream = _openseeker_truncate_to_tokens(text, 10)
        assert ours != upstream
        assert ours == upstream + "..."

    def test_uses_cl100k_encoding(self):
        """Must use cl100k_base (same as OpenSeeker)."""
        assert TOKEN_ENCODING.name == "cl100k_base"

    def test_allows_special_tokens(self):
        """Encoding must allow special tokens so Jina markdown doesn't crash.

        Upstream uses `encoding.encode(text, allowed_special='all')` explicitly
        because Jina output may contain substrings like `<|endofprompt|>`.
        """
        toolkit = WebScraperToolkit(token_budget=100)
        text_with_special = "before <|endoftext|> after"
        # Should not raise.
        result = toolkit.truncate_text(text_with_special)
        assert "before" in result


class TestPromptTemplate:
    """Structural parity of the extraction prompt."""

    def test_contains_webpage_content_section(self):
        assert "Webpage Content" in SUMMARY_PROMPT_TEMPLATE

    def test_contains_user_goal_section(self):
        assert "User Goal" in SUMMARY_PROMPT_TEMPLATE

    def test_contains_three_task_guidelines(self):
        """Guideline sections drive the rationale/evidence/summary fields."""
        assert "Content Scanning" in SUMMARY_PROMPT_TEMPLATE
        assert "Key Extraction" in SUMMARY_PROMPT_TEMPLATE
        assert "Summary Output" in SUMMARY_PROMPT_TEMPLATE

    def test_fixes_rationale_typo(self):
        """We fix upstream's 'Rational' → 'Rationale'. Lock it in."""
        assert "Rationale" in SUMMARY_PROMPT_TEMPLATE
        assert "Content Scanning for Rational:" not in SUMMARY_PROMPT_TEMPLATE

    def test_drops_json_format_footer(self):
        """Structured output replaces the upstream 'Final Output Format ... feilds' line."""
        assert "feilds" not in SUMMARY_PROMPT_TEMPLATE
        assert "JSON format" not in SUMMARY_PROMPT_TEMPLATE

    def test_formats_with_content_and_goal(self):
        rendered = SUMMARY_PROMPT_TEMPLATE.format(content="PAGE", goal="GOAL")
        assert "PAGE" in rendered
        assert "GOAL" in rendered


class TestWebPageSummarySchema:
    """Pydantic schema fields for structured output."""

    def test_has_three_fields(self):
        assert set(WebPageSummary.model_fields) == {"rationale", "evidence", "summary"}

    def test_rationale_fixed_from_upstream(self):
        """Upstream used misspelled `rational`; we use `rationale`."""
        assert "rationale" in WebPageSummary.model_fields
        assert "rational" not in WebPageSummary.model_fields

    def test_all_fields_required(self):
        for field in WebPageSummary.model_fields.values():
            assert field.is_required()


class TestScrapeResultTemplate:
    """Output format parity with OpenSeeker's formatted string."""

    def test_contains_url_and_goal_header(self):
        rendered = RESULT_TEMPLATE.format(url="http://x", goal="G", evidence="E", summary="S")
        assert "http://x" in rendered
        assert "G" in rendered
        assert "The useful information in" in rendered

    def test_contains_evidence_and_summary_sections(self):
        rendered = RESULT_TEMPLATE.format(url="http://x", goal="G", evidence="E", summary="S")
        assert "Evidence in page:" in rendered
        assert "Summary:" in rendered
        assert "E" in rendered
        assert "S" in rendered

    def test_structure_matches_upstream_modulo_trailing_spaces(self):
        """Upstream has trailing spaces after 'as follows:', 'Evidence in page:', 'Summary:'; we don't.

        Strip trailing spaces from each line and compare — the layout should match.
        """
        ours = RESULT_TEMPLATE.format(url="URL", goal="GOAL", evidence="EV", summary="SUM")
        upstream = (
            "The useful information in URL for user goal GOAL as follows: \n\n"
            "Evidence in page: \nEV\n\n"
            "Summary: \nSUM\n\n"
        )

        def _strip_trailing(text: str) -> str:
            # Strip per-line trailing spaces and any overall trailing blank lines.
            return "\n".join(line.rstrip() for line in text.splitlines()).rstrip()

        assert _strip_trailing(ours) == _strip_trailing(upstream)


@pytest.fixture(autouse=True)
def _jina_api_key(monkeypatch):
    """`scrape` is gated by `@requires_env("JINA_API_KEY")`; stub it for unit tests."""
    monkeypatch.setenv("JINA_API_KEY", "test-key")


class TestScrapeBehavior:
    """Behavioral tests for the `scrape` @tool with mocked fetch + summarize."""

    @pytest.mark.asyncio
    async def test_single_url_no_summarizer_returns_truncated_content(self):
        toolkit = WebScraperToolkit(token_budget=100)
        with patch.object(toolkit, "fetch_html", new=AsyncMock(return_value="# hello world")):
            result = await toolkit.scrape("http://x", goal="G")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_single_url_with_summarizer_formats_structured(self):
        factory = MagicMock()
        toolkit = WebScraperToolkit(summarizer_model_factory=factory)
        summary = WebPageSummary(rationale="R", evidence="E1", summary="S1")
        with patch.object(toolkit, "fetch_html", new=AsyncMock(return_value="raw page")):
            with patch.object(toolkit, "summarize", new=AsyncMock(return_value=summary)):
                result = await toolkit.scrape("http://x", goal="find Y")
        assert "http://x" in result
        assert "find Y" in result
        assert "E1" in result
        assert "S1" in result
        # Rationale is not emitted in the final user-facing output (matches upstream).
        assert "R" not in result or "Rationale" not in result

    @pytest.mark.asyncio
    async def test_single_url_fetch_failure_returns_failure_template(self):
        factory = MagicMock()
        toolkit = WebScraperToolkit(summarizer_model_factory=factory)
        with patch.object(toolkit, "fetch_html", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await toolkit.scrape("http://x", goal="G")
        assert "could not be accessed" in result
        assert "http://x" in result

    @pytest.mark.asyncio
    async def test_single_url_summarize_returns_none_returns_failure_template(self):
        factory = MagicMock()
        toolkit = WebScraperToolkit(summarizer_model_factory=factory)
        with patch.object(toolkit, "fetch_html", new=AsyncMock(return_value="raw")):
            with patch.object(toolkit, "summarize", new=AsyncMock(return_value=None)):
                result = await toolkit.scrape("http://x", goal="G")
        assert "could not be processed" in result

    @pytest.mark.asyncio
    async def test_batch_joined_with_triple_dash_separator(self):
        """Matches OpenSeeker's `\\n---\\n` join."""
        factory = MagicMock()
        toolkit = WebScraperToolkit(summarizer_model_factory=factory)
        summary = WebPageSummary(rationale="R", evidence="E", summary="S")
        with patch.object(toolkit, "fetch_html", new=AsyncMock(return_value="raw")):
            with patch.object(toolkit, "summarize", new=AsyncMock(return_value=summary)):
                result = await toolkit.scrape(["http://a", "http://b"], goal="G")
        assert "\n---\n" in result
        assert "http://a" in result
        assert "http://b" in result


class TestFetchHtmlRetry:
    """`fetch_html` retry behavior — treats empty body as retryable."""

    @pytest.mark.asyncio
    async def test_empty_body_triggers_retry_then_succeeds(self):
        toolkit = WebScraperToolkit()
        responses = ["", "  ", "real content"]
        call_count = 0

        class _FakeResponse:
            def __init__(self, text_value):
                self._text = text_value

            def raise_for_status(self):
                return None

            async def text(self):
                return self._text

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class _FakeSession:
            def get(self, url, headers=None):
                nonlocal call_count
                resp = _FakeResponse(responses[call_count])
                call_count += 1
                return resp

        with patch.object(toolkit, "_get_session", return_value=_FakeSession()):
            with patch("asyncio.sleep", new=AsyncMock()):
                result = await toolkit.fetch_html("http://x", max_retries=5, retry_delay=0)

        assert result == "real content"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_attempts_fail_raises_last_exception(self):
        toolkit = WebScraperToolkit()

        class _FakeResponse:
            def raise_for_status(self):
                raise RuntimeError("500")

            async def text(self):
                return ""

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

        class _FakeSession:
            def get(self, url, headers=None):
                return _FakeResponse()

        with patch.object(toolkit, "_get_session", return_value=_FakeSession()):
            with patch("asyncio.sleep", new=AsyncMock()):
                with pytest.raises(RuntimeError, match="500"):
                    await toolkit.fetch_html("http://x", max_retries=3, retry_delay=0)


class TestScrapeRequiresJinaApiKey:
    """`scrape` is gated by `@requires_env("JINA_API_KEY")`."""

    @pytest.mark.asyncio
    async def test_missing_jina_api_key_returns_error_string(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        toolkit = WebScraperToolkit()
        result = await toolkit.scrape("http://x", goal="G")
        assert "JINA_API_KEY" in result


class TestScraperDefaults:
    """Defaults are intentional choices that diverge from upstream."""

    def test_token_budget_default_diverges_from_upstream(self):
        """We default to 20K; upstream uses 95K. Callers opt in for OpenSeeker parity."""
        toolkit = WebScraperToolkit()
        assert toolkit.token_budget == 20_000
        assert toolkit.token_budget != 95_000

    def test_fetch_html_default_retries_match_upstream_outer_loop(self):
        """Upstream's `html_readpage_jina` does 8 outer attempts."""
        import inspect

        sig = inspect.signature(WebScraperToolkit.fetch_html)
        assert sig.parameters["max_retries"].default == 8
