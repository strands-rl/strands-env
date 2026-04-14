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

"""Unit tests for model factory functions."""

from unittest.mock import MagicMock, patch

import boto3
import pytest
from strands_sglang import SGLangClient

from strands_env.core.models import (
    DEFAULT_SAMPLING_PARAMS,
    bedrock_model_factory,
    openai_model_factory,
    sglang_model_factory,
)

# ---------------------------------------------------------------------------
# sglang_model_factory
# ---------------------------------------------------------------------------


class TestSGLangModelFactory:
    def test_each_call_creates_new_instance(self):
        factory = sglang_model_factory(
            tokenizer=MagicMock(),
            client=MagicMock(spec=SGLangClient),
        )
        model1 = factory()
        model2 = factory()
        assert model1 is not model2


# ---------------------------------------------------------------------------
# bedrock_model_factory
# ---------------------------------------------------------------------------


class TestBedrockModelFactory:
    @patch("strands_env.core.models.BedrockModel")
    def test_remaps_max_new_tokens(self, mock_bedrock_cls):
        factory = bedrock_model_factory(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            boto_session=MagicMock(spec=boto3.Session),
            sampling_params={"max_new_tokens": 2048, "temperature": 0.7},
        )
        factory()

        call_kwargs = mock_bedrock_cls.call_args[1]
        assert "max_tokens" in call_kwargs
        assert "max_new_tokens" not in call_kwargs
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.7

    @patch("strands_env.core.models.BedrockModel")
    def test_does_not_mutate_default_params(self, mock_bedrock_cls):
        original = dict(DEFAULT_SAMPLING_PARAMS)
        bedrock_model_factory(
            model_id="test",
            boto_session=MagicMock(spec=boto3.Session),
        )
        assert DEFAULT_SAMPLING_PARAMS == original

    @patch("strands_env.core.models.BedrockModel")
    def test_shared_client_across_instances(self, mock_bedrock_cls):
        """All models from the same factory should share a single boto3 client."""
        mock_client = MagicMock()
        mock_bedrock_cls.return_value.client = mock_client

        factory = bedrock_model_factory(
            model_id="test",
            boto_session=MagicMock(spec=boto3.Session),
        )
        model1 = factory()
        model2 = factory()
        assert model1.client is model2.client
        assert model1.client is mock_client


# ---------------------------------------------------------------------------
# openai_model_factory
# ---------------------------------------------------------------------------


class TestOpenAIModelFactory:
    @patch("strands_env.core.models.OpenAIModel")
    def test_remaps_max_new_tokens(self, mock_openai_cls):
        factory = openai_model_factory(
            model_id="gpt-4o",
            sampling_params={"max_new_tokens": 4096, "temperature": 0.5},
        )
        factory()

        call_kwargs = mock_openai_cls.call_args[1]
        assert call_kwargs["params"]["max_tokens"] == 4096
        assert "max_new_tokens" not in call_kwargs["params"]

    @patch("strands_env.core.models.OpenAIModel")
    def test_does_not_mutate_default_params(self, mock_openai_cls):
        original = dict(DEFAULT_SAMPLING_PARAMS)
        openai_model_factory(model_id="gpt-4o")
        assert DEFAULT_SAMPLING_PARAMS == original


# ---------------------------------------------------------------------------
# kimi_model_factory
# ---------------------------------------------------------------------------


class TestKimiModelFactory:
    @pytest.fixture(autouse=True)
    def _require_litellm(self):
        pytest.importorskip("litellm")

    @pytest.fixture(autouse=True)
    def _set_moonshot_key(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    def test_remaps_max_new_tokens(self):
        from strands_env.core.models import kimi_model_factory

        factory = kimi_model_factory(sampling_params={"max_new_tokens": 4096})
        model = factory()
        assert model.get_config()["params"]["max_tokens"] == 4096
        assert "max_new_tokens" not in model.get_config()["params"]

    def test_preserves_reasoning_content(self):
        """KimiModel._format_regular_messages preserves reasoningContent as top-level field."""
        from strands_env.core.models import _get_kimi_model_class

        kimi_model_cls = _get_kimi_model_class()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "Let me think..."}}},
                    {"text": "The answer is 42."},
                ],
            }
        ]
        formatted = kimi_model_cls._format_regular_messages(messages)
        assert len(formatted) == 1
        assert formatted[0]["reasoning_content"] == "Let me think..."
        assert all("reasoning" not in str(c) for c in formatted[0]["content"])
