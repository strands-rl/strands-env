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

"""Unit tests for OpenSeekerVisitToolkit pure-logic methods."""

from strands_env.tools.openseeker_visit import (
    OpenSeekerVisitToolkit,
    _is_valid_content,
    _truncate_to_tokens,
)


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        assert _truncate_to_tokens("hello", max_tokens=100) == "hello"

    def test_truncates_long_text(self):
        long = "word " * 50_000
        result = _truncate_to_tokens(long, max_tokens=100)
        assert len(result) < len(long)

    def test_zero_tokens_returns_empty(self):
        assert _truncate_to_tokens("hello world", max_tokens=0) == ""


class TestIsValidContent:
    def test_valid(self):
        assert _is_valid_content("some page content") is True

    def test_empty(self):
        assert _is_valid_content("") is False

    def test_failed(self):
        assert _is_valid_content("[visit] Failed to read page.") is False

    def test_empty_content_marker(self):
        assert _is_valid_content("[visit] Empty content.") is False


class TestParseSummary:
    def test_valid_json(self):
        raw = '{"rational": "r", "evidence": "e", "summary": "s"}'
        assert OpenSeekerVisitToolkit._parse_summary(raw) is not None

    def test_json_in_code_block(self):
        raw = '```json\n{"evidence": "e", "summary": "s"}\n```'
        assert OpenSeekerVisitToolkit._parse_summary(raw)["evidence"] == "e"

    def test_json_embedded_in_text(self):
        raw = 'Here is the result: {"evidence": "e", "summary": "s"} end.'
        assert OpenSeekerVisitToolkit._parse_summary(raw)["summary"] == "s"

    def test_invalid_returns_none(self):
        assert OpenSeekerVisitToolkit._parse_summary("not json") is None

    def test_empty_returns_none(self):
        assert OpenSeekerVisitToolkit._parse_summary("") is None
