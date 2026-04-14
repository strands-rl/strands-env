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

"""Shared fixtures for integration tests.

All tests in this directory require a running SGLang server.
The model ID is auto-detected from the server via /get_model_info.
Tests are skipped automatically if the server is not reachable.

Configuration (priority: CLI > env var > default):
    pytest --sglang-base-url=http://localhost:30000 --tool-parser=hermes
    SGLANG_BASE_URL=http://... TOOL_PARSER=hermes pytest tests/integration/
"""

import pytest
from strands_sglang import SGLangClient
from strands_sglang.tool_parsers import get_tool_parser
from transformers import AutoTokenizer

from strands_env.core.models import DEFAULT_SAMPLING_PARAMS, sglang_model_factory
from strands_env.core.types import StepResult, TerminationReason
from strands_env.utils.sglang import check_server_health, get_model_id

# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_successful_step(result: StepResult) -> None:
    """Assert that a step completed successfully with messages and a text response."""
    assert result.termination_reason == TerminationReason.TASK_COMPLETE
    assert result.observation.messages
    assert result.observation.final_response


def assert_token_observation(result: StepResult) -> None:
    """Assert that token-level observation has valid structure."""
    tokens = result.observation.tokens
    assert tokens is not None
    assert tokens.prompt_length > 0
    assert len(tokens.rollout_token_ids) > 0
    assert len(tokens.loss_mask) == len(tokens.token_ids)
    assert len(tokens.logprobs) == len(tokens.token_ids)
    assert any(lp is not None for lp in tokens.rollout_logprobs)


def assert_token_usage(result: StepResult) -> None:
    """Assert that input/output token usage dicts have expected structure."""
    metrics = result.observation.metrics
    assert metrics["model_calls"] >= 1
    for key in ("input_tokens", "output_tokens"):
        for subkey in ("total", "max", "mean", "min"):
            assert metrics[key][subkey] > 0


# Mark all tests in this directory as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def sglang_base_url(request):
    """Get SGLang server URL from pytest config."""
    return request.config.getoption("--sglang-base-url")


@pytest.fixture(scope="session")
def tool_parser_name(request):
    """Get tool parser name from CLI option."""
    return request.config.getoption("--tool-parser")


@pytest.fixture
async def sglang_client(sglang_base_url):
    """Fresh SGLang client per test to avoid event-loop affinity issues with aiohttp."""
    try:
        check_server_health(sglang_base_url)
    except ConnectionError:
        pytest.skip(f"SGLang server not reachable at {sglang_base_url}")
    client = SGLangClient(sglang_base_url)
    yield client
    await client.close()


@pytest.fixture(scope="session")
def sglang_model_id(sglang_base_url):
    """Auto-detect model ID from the running SGLang server."""
    return get_model_id(sglang_base_url)


@pytest.fixture(scope="session")
def tokenizer(sglang_model_id):
    """Load tokenizer for the detected model."""
    return AutoTokenizer.from_pretrained(sglang_model_id)


@pytest.fixture
def model_factory(tokenizer, sglang_client, tool_parser_name):
    """Model factory for Environment integration tests."""
    return sglang_model_factory(
        tokenizer=tokenizer,
        client=sglang_client,
        tool_parser=get_tool_parser(tool_parser_name),
        sampling_params=DEFAULT_SAMPLING_PARAMS,
    )
