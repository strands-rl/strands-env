"""Unit tests for WebSearchToolkit pure-logic methods."""

from strands_env.environments.web_search.tools.search import WebSearchToolkit

# ---------------------------------------------------------------------------
# _apply_blocked_domains
# ---------------------------------------------------------------------------


class TestApplyBlockedDomains:
    def test_appends_exclusions(self):
        result = WebSearchToolkit.apply_blocked_domains("python asyncio", ["example.com", "spam.org"])
        assert result == "python asyncio -site:example.com -site:spam.org"

    def test_empty_blocked_domains(self):
        result = WebSearchToolkit.apply_blocked_domains("python asyncio", [])
        assert result == "python asyncio"

    def test_no_blocked_domains(self):
        result = WebSearchToolkit.apply_blocked_domains("query", [])
        assert result == "query"


# ---------------------------------------------------------------------------
# format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    def test_formats_numbered_list(self):
        items = [
            {"title": "Result One", "link": "https://one.com", "snippet": "First snippet."},
            {"title": "Result Two", "link": "https://two.com", "snippet": "Second snippet."},
        ]
        result = WebSearchToolkit.format_results(items)
        assert "1. Result One (https://one.com):" in result
        assert "First snippet." in result
        assert "2. Result Two (https://two.com):" in result
        assert "Second snippet." in result

    def test_empty_results(self):
        result = WebSearchToolkit.format_results([])
        assert result == "No results found."

    def test_missing_fields_use_defaults(self):
        items = [{}]
        result = WebSearchToolkit.format_results(items)
        assert "No title available." in result
        assert "No URL available." in result
        assert "No snippet available." in result
