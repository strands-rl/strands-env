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

"""Unit tests for Environment."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands_env.core.environment import Environment
from strands_env.core.types import (
    RewardResult,
    Task,
    TaskContext,
    TerminationReason,
)

from .conftest import mock_agent, mock_event_loop_metrics

# ---------------------------------------------------------------------------
# Constructor — system prompt loading
# ---------------------------------------------------------------------------


class TestEnvironmentInit:
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
# rollout()
# ---------------------------------------------------------------------------


class TestRollout:
    @patch("strands_env.core.environment.Agent")
    async def test_successful_rollout(self, mock_agent_cls, env):
        """A successful agent invocation returns TASK_COMPLETE."""
        conversation_history = [{"role": "user", "content": [{"text": "earlier"}]}]
        mock_agent_cls.return_value = mock_agent(
            messages=conversation_history + [{"role": "assistant", "content": [{"text": "answer"}]}],
        )

        task = Task(
            message="What is 2+2?",
            context=TaskContext(conversation_history=conversation_history),
        )
        result = await env.rollout(task)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.metrics["message_count"] == 1
        assert result.reward is None

    @patch("strands_env.core.environment.Agent")
    async def test_rollout_with_agent_error(self, mock_agent_cls, env):
        """An unrecognized exception maps to UNCLASSIFIED_ERROR."""
        agent = mock_agent()
        agent.invoke_async.side_effect = RuntimeError("boom")
        mock_agent_cls.return_value = agent

        task = Task(message="Do something")
        result = await env.rollout(task)

        assert result.termination_reason == TerminationReason.UNCLASSIFIED_ERROR

    @patch("strands_env.core.environment.Agent")
    async def test_rollout_with_reward_fn(self, mock_agent_cls, model_factory):
        """Reward function is called when provided."""
        mock_agent_cls.return_value = mock_agent(
            messages=[{"role": "assistant", "content": [{"text": "4"}]}],
        )

        reward_fn = MagicMock()
        reward_fn.compute = AsyncMock(return_value=RewardResult(reward=1.0))
        env = Environment(model_factory=model_factory, reward_fn=reward_fn)

        task = Task(message="What is 2+2?", context=TaskContext(ground_truth="4"))
        result = await env.rollout(task)

        reward_fn.compute.assert_awaited_once()
        assert result.reward.reward == 1.0
        # Regression: timing must be written via `result.metrics`, not the
        # local `metrics` dict — Pydantic v2 rebuilds the dict at validation, so
        # the local var is decoupled from the model after `RolloutResult(...)`.
        assert "reward_latency_s" in result.metrics
        assert result.metrics["reward_latency_s"] >= 0.0

    @patch("strands_env.core.environment.Agent")
    async def test_rollout_with_dict_message(self, mock_agent_cls, env):
        """Task.message can be a dict with 'content' key."""
        mock_agent_cls.return_value = mock_agent(
            messages=[{"role": "assistant", "content": [{"text": "answer"}]}],
        )

        task = Task(message={"role": "user", "content": [{"text": "hello"}]})
        result = await env.rollout(task)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        # Agent.invoke_async should receive the content list, not the full dict
        mock_agent_cls.return_value.invoke_async.assert_awaited_once_with([{"text": "hello"}])

    @patch("strands_env.core.environment.Agent")
    async def test_rollout_messages_sliced(self, mock_agent_cls, env):
        """result.messages only contains messages added during the rollout."""
        history = [
            {"role": "user", "content": [{"text": "msg1"}]},
            {"role": "assistant", "content": [{"text": "resp1"}]},
        ]
        new_messages = [
            {"role": "user", "content": [{"text": "msg2"}]},
            {"role": "assistant", "content": [{"text": "resp2"}]},
        ]
        mock_agent_cls.return_value = mock_agent(messages=history + new_messages)

        task = Task(message="msg2", context=TaskContext(conversation_history=history))
        result = await env.rollout(task)

        assert result.metrics["message_count"] == 2
        assert result.messages == new_messages

    @patch("strands_env.core.environment.Agent")
    async def test_rollout_records_tool_limiter_counts(self, mock_agent_cls, env):
        """Tool limiter iteration/call/cancelled counts appear in metrics."""
        mock_agent_cls.return_value = mock_agent(
            messages=[{"role": "assistant", "content": [{"text": "done"}]}],
        )

        result = await env.rollout(Task(message="test"))

        assert "tool_iters" in result.metrics
        assert "tool_calls" in result.metrics
        assert "cancelled_tool_calls" in result.metrics


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
        metrics = mock_event_loop_metrics()
        metrics.cycle_count = 3
        metrics.agent_invocations[0].cycles = cycles
        metrics.cycle_durations = [0.8, 0.9, 0.8]

        result = env.compute_metrics(metrics)

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

        metrics = mock_event_loop_metrics()
        metrics.cycle_count = 2
        metrics.cycle_durations = [0.5]
        metrics.tool_metrics = {"calculator": tool_metric}

        result = env.compute_metrics(metrics, tool_parse_errors={"calculator": 2})

        assert result["per_tool_metrics"]["calculator"]["calls"] == 5
        assert result["per_tool_metrics"]["calculator"]["successes"] == 4
        assert result["per_tool_metrics"]["calculator"]["errors"] == 1
        assert result["per_tool_metrics"]["calculator"]["parse_errors"] == 2
        assert result["per_tool_metrics"]["calculator"]["latency_s"] == 1.2345

    def test_cache_hit_rate(self, env):
        """Non-zero cache reads produce a cache_hit_rate."""
        cycles = [self._make_cycle(100, 20, cache_read_input_tokens=40)]
        metrics = mock_event_loop_metrics()
        metrics.cycle_count = 1
        metrics.agent_invocations[0].cycles = cycles
        metrics.cycle_durations = [0.5]

        result = env.compute_metrics(metrics)

        assert result["cache_hit_rate"] == pytest.approx(0.4)
        assert result["cache_read_input_tokens"]["total"] == 40

    def test_cache_read_zero_means_no_summary(self, env):
        """All cache reads are 0 → cache_read_input_tokens is None (not a summary of zeros)."""
        cycles = [self._make_cycle(100, 20, cache_read_input_tokens=0)]
        metrics = mock_event_loop_metrics()
        metrics.cycle_count = 1
        metrics.agent_invocations[0].cycles = cycles
        metrics.cycle_durations = [0.5]

        result = env.compute_metrics(metrics)

        # any(cache_read_counts) is False when all are 0 → no summary
        assert result["cache_read_input_tokens"] is None

    def test_cache_hit_rate_zero_inputs(self, env):
        """Zero input tokens → cache_hit_rate is None."""
        metrics = mock_event_loop_metrics()
        metrics.cycle_count = 0
        metrics.agent_invocations = []
        metrics.cycle_durations = []

        result = env.compute_metrics(metrics)

        assert result["cache_hit_rate"] is None


# ---------------------------------------------------------------------------
# Config type-hint resolution
# ---------------------------------------------------------------------------


class TestEnvironmentConfigTypeHints:
    """`EnvironmentConfig` annotations must resolve at runtime.

    Subclass configs are embedded as Pydantic fields by benchmark task contexts
    (e.g. `config: HarborConfig`), so Pydantic resolves the hints at runtime. A
    TYPE_CHECKING-only import in an annotation (e.g. `AttributeValue`) breaks
    that with a `NameError`.
    """

    def test_type_hints_resolve(self):
        """get_type_hints succeeds — no unresolved forward refs."""
        import typing

        from strands_env.core.environment import EnvironmentConfig

        hints = typing.get_type_hints(EnvironmentConfig)
        assert "trace_attributes" in hints
