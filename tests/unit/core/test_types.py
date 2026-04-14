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

"""Unit tests for core types."""

from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
    MaxTokensReachedException,
)
from strands_sglang import MaxToolCallsReachedError, MaxToolIterationsReachedError, TokenManager

from strands_env.core.types import (
    Action,
    Observation,
    RewardResult,
    StepResult,
    TaskContext,
    TerminationReason,
    TokenObservation,
)

# ---------------------------------------------------------------------------
# TaskContext
# ---------------------------------------------------------------------------


class TestTaskContext:
    def test_defaults(self):
        ctx = TaskContext()
        assert ctx.ground_truth is None
        assert ctx.conversation_history == []

    def test_extra_fields(self):
        ctx = TaskContext(ground_truth="42", difficulty=3, tags=["math"])
        assert ctx.ground_truth == "42"
        assert ctx.difficulty == 3
        assert ctx.tags == ["math"]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TestAction:
    def test_string_message(self):
        action = Action(message="What is 2+2?")
        assert action.message == "What is 2+2?"
        assert action.task_context.ground_truth is None

    def test_with_context(self):
        ctx = TaskContext(ground_truth="4")
        action = Action(message="What is 2+2?", task_context=ctx)
        assert action.task_context.ground_truth == "4"


# ---------------------------------------------------------------------------
# TokenObservation
# ---------------------------------------------------------------------------


class TestTokenObservation:
    def test_rollout_slicing(self):
        obs = TokenObservation(
            token_ids=[10, 20, 30, 40, 50],
            prompt_length=2,
            loss_mask=[0, 0, 1, 1, 1],
            logprobs=[None, None, -0.5, -0.3, -0.1],
        )
        assert obs.initial_prompt_token_ids == [10, 20]
        assert obs.rollout_token_ids == [30, 40, 50]
        assert obs.rollout_loss_mask == [1, 1, 1]
        assert obs.rollout_logprobs == [-0.5, -0.3, -0.1]

    def test_from_token_manager_empty(self):
        tm = TokenManager()
        assert TokenObservation.from_token_manager(tm) is None

    def test_from_token_manager(self):
        tm = TokenManager()
        tm.add_prompt([1, 2, 3])
        tm.add_response([4, 5], logprobs=[-0.1, -0.2])
        obs = TokenObservation.from_token_manager(tm)
        assert obs is not None
        assert obs.token_ids == [1, 2, 3, 4, 5]
        assert obs.prompt_length == 3
        assert obs.rollout_token_ids == [4, 5]

    def test_from_token_manager_uses_initial_prompt(self):
        """Verify prompt_length is derived from the first segment."""
        tm = TokenManager()
        tm.add_prompt([10, 20])
        tm.add_response([30])
        tm.add_prompt([40])
        tm.add_response([50])
        obs = TokenObservation.from_token_manager(tm)
        assert obs.prompt_length == 2
        assert obs.initial_prompt_token_ids == [10, 20]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TestObservation:
    def test_final_response_from_assistant(self):
        messages = [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}, {"text": "world"}]},
        ]
        obs = Observation(messages=messages)
        assert obs.final_response == "world"

    def test_final_response_strips_think_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "<think>reasoning here</think>The actual answer"}],
            },
        ]
        obs = Observation(messages=messages)
        assert obs.final_response == "The actual answer"

    def test_final_response_strips_nested_think_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "<think>first</think>middle<think>second</think>final answer"}],
            },
        ]
        obs = Observation(messages=messages)
        assert obs.final_response == "final answer"

    def test_final_response_no_assistant(self):
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        obs = Observation(messages=messages)
        assert obs.final_response is None

    def test_final_response_empty(self):
        obs = Observation()
        assert obs.final_response is None

    def test_final_response_skips_non_text_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"name": "calc", "toolUseId": "123", "input": {}}},
                    {"text": "result is 4"},
                ],
            },
        ]
        obs = Observation(messages=messages)
        assert obs.final_response == "result is 4"


# ---------------------------------------------------------------------------
# TerminationReason
# ---------------------------------------------------------------------------


class TestTerminationReason:
    def test_none_error_is_task_complete(self):
        assert TerminationReason.from_error(None) == TerminationReason.TASK_COMPLETE

    def test_max_tool_iterations(self):
        error = EventLoopException(Exception())
        error.__cause__ = MaxToolIterationsReachedError(10)
        assert TerminationReason.from_error(error) == TerminationReason.MAX_TOOL_ITERATIONS_REACHED

    def test_max_tool_calls(self):
        error = EventLoopException(Exception())
        error.__cause__ = MaxToolCallsReachedError(5)
        assert TerminationReason.from_error(error) == TerminationReason.MAX_TOOL_CALLS_REACHED

    def test_max_tokens(self):
        error = EventLoopException(Exception())
        error.__cause__ = MaxTokensReachedException("max tokens reached")
        assert TerminationReason.from_error(error) == TerminationReason.MAX_TOKENS_REACHED

    def test_context_window_overflow(self):
        error = EventLoopException(Exception())
        error.__cause__ = ContextWindowOverflowException(Exception("context window exceeded"))
        assert TerminationReason.from_error(error) == TerminationReason.CONTEXT_WINDOW_OVERFLOW

    def test_timeout(self):
        class ReadTimeoutError(Exception):
            pass

        error = EventLoopException(Exception())
        error.__cause__ = ReadTimeoutError()
        assert TerminationReason.from_error(error) == TerminationReason.TIMEOUT

    def test_timeout_in_cause_chain(self):
        class ConnectTimeoutError(Exception):
            pass

        inner = ConnectTimeoutError()
        outer = RuntimeError("wrapper")
        outer.__cause__ = inner
        assert TerminationReason.from_error(outer) == TerminationReason.TIMEOUT

    def test_generic_error(self):
        error = EventLoopException(Exception())
        error.__cause__ = ValueError("something broke")
        assert TerminationReason.from_error(error) == TerminationReason.UNCLASSIFIED_ERROR

    def test_non_event_loop_exception(self):
        error = RuntimeError("direct error")
        assert TerminationReason.from_error(error) == TerminationReason.UNCLASSIFIED_ERROR


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_defaults(self):
        obs = Observation()
        result = StepResult(observation=obs)
        assert result.reward is None
        assert result.termination_reason == TerminationReason.NOT_TERMINATED

    def test_with_reward(self):
        obs = Observation()
        reward = RewardResult(reward=1.0, info={"exact_match": True})
        result = StepResult(observation=obs, reward=reward)
        assert result.reward.reward == 1.0
        assert result.reward.info["exact_match"] is True
