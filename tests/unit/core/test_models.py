from unittest.mock import MagicMock, patch

import boto3
import pytest
from strands_sglang import SGLangClient

from strands_env.core.models import (
    DEFAULT_SAMPLING_PARAMS,
    ModelConfig,
    bedrock_mantle_model_factory,
    bedrock_model_factory,
    build_model_factory,
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
        assert original == DEFAULT_SAMPLING_PARAMS

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
# bedrock_mantle_model_factory
# ---------------------------------------------------------------------------


class TestBedrockMantleModelFactory:
    def _patch_model(self):
        """Patch the OpenAIResponsesModel constructor imported into models.py."""
        responses_cls = MagicMock(name="OpenAIResponsesModel")
        return responses_cls, patch("strands_env.core.models.OpenAIResponsesModel", responses_cls)

    def test_builds_responses_model_with_mantle_config(self):
        responses_cls, patch_model = self._patch_model()
        with patch_model:
            factory = bedrock_mantle_model_factory(
                model_id="openai.gpt-5.4-2026-03-05",
                region="us-east-2",
                sampling_params={"max_new_tokens": 16384},
                reasoning={"effort": "high"},
            )
            factory()

        call_kwargs = responses_cls.call_args[1]
        assert call_kwargs["model_id"] == "openai.gpt-5.4-2026-03-05"
        # Base URL + SigV4 token are derived by the SDK from bedrock_mantle_config; we only pass region.
        assert call_kwargs["bedrock_mantle_config"] == {"region": "us-east-2"}
        assert "client_args" not in call_kwargs
        # 'stateful' is not passed: SDK default (False) applies, matching the other backends,
        # so the SDK never clears agent.messages.
        assert "stateful" not in call_kwargs
        # max_new_tokens -> max_output_tokens; reasoning forwarded.
        assert call_kwargs["params"]["max_output_tokens"] == 16384
        assert "max_new_tokens" not in call_kwargs["params"]
        assert call_kwargs["params"]["reasoning"] == {"effort": "high"}

    def test_does_not_mutate_default_params(self):
        original = dict(DEFAULT_SAMPLING_PARAMS)
        _, patch_model = self._patch_model()
        with patch_model:
            bedrock_mantle_model_factory(model_id="openai.gpt-5.4-2026-03-05")
        assert original == DEFAULT_SAMPLING_PARAMS

    def test_build_model_factory_requires_model_id(self):
        with pytest.raises(ValueError, match="bedrock-mantle backend requires"):
            build_model_factory(ModelConfig(backend="bedrock-mantle"))


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
        assert original == DEFAULT_SAMPLING_PARAMS
