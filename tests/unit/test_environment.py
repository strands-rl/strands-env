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

"""Unit tests for Environment."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands_sglang import TokenManager

from strands_env.core.environment import Environment
from strands_env.core.types import (
    Action,
    RewardResult,
    TaskContext,
    TerminationReason,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.token_manager = TokenManager()
    return model


@pytest.fixture
def model_factory(mock_model):
    return lambda: mock_model


@pytest.fixture
def env(model_factory):
    return Environment(model_factory=model_factory)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestEnvironmentInit:
    def test_defaults(self, model_factory):
        env = Environment(model_factory=model_factory)
        assert env.max_tool_iters is None
        assert env.max_tool_calls is None
        assert env.max_parallel_tool_calls is None
        assert env.verbose is False
        assert env.system_prompt is None
        assert env.reward_fn is None

    def test_custom_system_prompt(self, model_factory):
        env = Environment(model_factory=model_factory, system_prompt="You are helpful.")
        assert env.system_prompt == "You are helpful."

    def test_system_prompt_from_file(self, model_factory, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("Be concise.")

        class MyEnv(Environment):
            default_system_prompt_path = prompt_file

        env = MyEnv(model_factory=model_factory)
        assert env.system_prompt == "Be concise."

    def test_explicit_prompt_overrides_file(self, model_factory, tmp_path):
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text("From file.")

        class MyEnv(Environment):
            default_system_prompt_path = prompt_file

        env = MyEnv(model_factory=model_factory, system_prompt="From arg.")
        assert env.system_prompt == "From arg."


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    @patch("strands_env.core.environment.Agent")
    async def test_successful_step(self, mock_agent_cls, env):
        """A successful agent invocation returns TASK_COMPLETE."""
        conversation_history = [{"role": "user", "content": [{"text": "earlier"}]}]
        agent_instance = MagicMock()
        agent_instance.invoke_async = AsyncMock()
        agent_instance.messages = conversation_history + [
            {"role": "assistant", "content": [{"text": "answer"}]},
        ]
        agent_instance.model.token_manager = TokenManager()
        agent_instance.event_loop_metrics = self._mock_event_loop_metrics()
        mock_agent_cls.return_value = agent_instance

        action = Action(
            message="What is 2+2?",
            task_context=TaskContext(conversation_history=conversation_history),
        )
        result = await env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.metrics["message_count"] == 1
        assert result.reward is None

    @patch("strands_env.core.environment.Agent")
    async def test_step_with_agent_error(self, mock_agent_cls, env):
        """An unrecognized exception maps to UNCLASSIFIED_ERROR."""
        agent_instance = MagicMock()
        agent_instance.invoke_async = AsyncMock(side_effect=RuntimeError("boom"))
        agent_instance.messages = []
        agent_instance.model.token_manager = TokenManager()
        agent_instance.event_loop_metrics = self._mock_event_loop_metrics()
        mock_agent_cls.return_value = agent_instance

        action = Action(message="Do something")
        result = await env.step(action)

        assert result.termination_reason == TerminationReason.UNCLASSIFIED_ERROR

    @patch("strands_env.core.environment.Agent")
    async def test_step_with_reward_fn(self, mock_agent_cls, model_factory):
        """Reward function is called when provided."""
        agent_instance = MagicMock()
        agent_instance.invoke_async = AsyncMock()
        agent_instance.messages = [{"role": "assistant", "content": [{"text": "4"}]}]
        agent_instance.model.token_manager = TokenManager()
        agent_instance.event_loop_metrics = self._mock_event_loop_metrics()
        mock_agent_cls.return_value = agent_instance

        reward_fn = MagicMock()
        reward_fn.compute = AsyncMock(return_value=RewardResult(reward=1.0))
        env = Environment(model_factory=model_factory, reward_fn=reward_fn)

        action = Action(message="What is 2+2?", task_context=TaskContext(ground_truth="4"))
        result = await env.step(action)

        reward_fn.compute.assert_awaited_once()
        assert result.reward.reward == 1.0

    @patch("strands_env.core.environment.Agent")
    async def test_step_messages_sliced(self, mock_agent_cls, env):
        """step_messages only contains messages added during the step."""
        history = [
            {"role": "user", "content": [{"text": "msg1"}]},
            {"role": "assistant", "content": [{"text": "resp1"}]},
        ]
        new_messages = [
            {"role": "user", "content": [{"text": "msg2"}]},
            {"role": "assistant", "content": [{"text": "resp2"}]},
        ]
        agent_instance = MagicMock()
        agent_instance.invoke_async = AsyncMock()
        agent_instance.messages = history + new_messages
        agent_instance.model.token_manager = TokenManager()
        agent_instance.event_loop_metrics = self._mock_event_loop_metrics()
        mock_agent_cls.return_value = agent_instance

        action = Action(message="msg2", task_context=TaskContext(conversation_history=history))
        result = await env.step(action)

        assert result.observation.metrics["message_count"] == 2
        assert result.observation.messages == new_messages

    @staticmethod
    def _mock_event_loop_metrics():
        cycle = MagicMock()
        cycle.usage = {"inputTokens": 10, "outputTokens": 5}
        invocation = MagicMock()
        invocation.cycles = [cycle]

        metrics = MagicMock()
        metrics.cycle_count = 1
        metrics.agent_invocations = [invocation]
        metrics.cycle_durations = [0.1]
        metrics.tool_metrics = {}
        return metrics


# ---------------------------------------------------------------------------
# compute_metrics()
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    @staticmethod
    def _make_cycle(input_tokens, output_tokens, cache_read_input_tokens=0):
        cycle = MagicMock()
        cycle.usage = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cacheReadInputTokens": cache_read_input_tokens,
        }
        return cycle

    def test_basic_metrics_without_cache(self, env):
        cycles = [self._make_cycle(30, 15), self._make_cycle(35, 20), self._make_cycle(35, 15)]
        invocation = MagicMock()
        invocation.cycles = cycles

        event_loop_metrics = MagicMock()
        event_loop_metrics.cycle_count = 3
        event_loop_metrics.agent_invocations = [invocation]
        event_loop_metrics.cycle_durations = [0.8, 0.9, 0.8]
        event_loop_metrics.tool_metrics = {}

        result = env.compute_metrics(event_loop_metrics)

        assert result["model_calls"] == 3
        assert result["input_tokens"]["total"] == 100
        assert result["output_tokens"]["total"] == 50
        assert result["model_latency_s"]["total"] == 2.5
        assert result["per_tool_metrics"] is None

    def test_with_tool_metrics(self, env):
        tool_metric = MagicMock()
        tool_metric.call_count = 5
        tool_metric.success_count = 4
        tool_metric.error_count = 1
        tool_metric.total_time = 1.2345

        invocation = MagicMock()
        invocation.cycles = [self._make_cycle(10, 5)]

        event_loop_metrics = MagicMock()
        event_loop_metrics.cycle_count = 2
        event_loop_metrics.agent_invocations = [invocation]
        event_loop_metrics.cycle_durations = [0.5]
        event_loop_metrics.tool_metrics = {"calculator": tool_metric}

        result = env.compute_metrics(event_loop_metrics, tool_parse_errors={"calculator": 2})

        assert result["per_tool_metrics"]["calculator"]["calls"] == 5
        assert result["per_tool_metrics"]["calculator"]["successes"] == 4
        assert result["per_tool_metrics"]["calculator"]["errors"] == 1
        assert result["per_tool_metrics"]["calculator"]["parse_errors"] == 2
        assert result["per_tool_metrics"]["calculator"]["latency_s"] == 1.2345

    def test_zero_values_preserved(self, env):
        event_loop_metrics = MagicMock()
        event_loop_metrics.cycle_count = 0
        event_loop_metrics.agent_invocations = []
        event_loop_metrics.cycle_durations = []
        event_loop_metrics.tool_metrics = {}

        result = env.compute_metrics(event_loop_metrics)

        assert result["model_calls"] == 0
        assert result["input_tokens"] is None
        assert result["model_latency_s"] is None

    def test_missing_latency(self, env):
        event_loop_metrics = MagicMock()
        event_loop_metrics.cycle_count = 1
        event_loop_metrics.agent_invocations = []
        event_loop_metrics.cycle_durations = []
        event_loop_metrics.tool_metrics = {}

        result = env.compute_metrics(event_loop_metrics)

        assert result["model_latency_s"] is None


# ---------------------------------------------------------------------------
# Overridable methods
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_get_tools_default_empty(self, env):
        assert env.get_tools() == []

    def test_get_hooks_default_empty(self, env):
        assert env.get_hooks() == []

    async def test_reset_is_noop(self, env):
        await env.reset()

    async def test_cleanup_is_noop(self, env):
        await env.cleanup()
