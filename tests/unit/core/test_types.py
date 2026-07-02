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
from strands_sglang import MaxMessagesReachedError, MaxToolCallsReachedError, MaxToolIterationsReachedError

from strands_env.core.types import (
    RewardResult,
    RolloutResult,
    Task,
    TaskContext,
    TerminationReason,
    extract_message_text,
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
# Task
# ---------------------------------------------------------------------------


class TestTask:
    def test_string_message(self):
        task = Task(message="What is 2+2?")
        assert task.message == "What is 2+2?"
        assert task.context.ground_truth is None

    def test_with_context(self):
        ctx = TaskContext(ground_truth="4")
        task = Task(message="What is 2+2?", context=ctx)
        assert task.context.ground_truth == "4"


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------


class TestExtractMessageText:
    def test_no_text_block_returns_empty(self):
        message = {"role": "assistant", "content": [{"toolUse": {"name": "calc", "toolUseId": "1", "input": {}}}]}
        assert extract_message_text(message) == ""

    def test_empty_content_returns_empty(self):
        assert extract_message_text({"role": "assistant", "content": []}) == ""

    def test_multiple_text_blocks_takes_last(self):
        message = {
            "role": "assistant",
            "content": [
                {"text": "Let me check."},
                {"toolUse": {"name": "calc", "toolUseId": "1", "input": {}}},
                {"text": "The answer is 4."},
            ],
        }
        assert extract_message_text(message) == "The answer is 4."

    def test_strips_think_block(self):
        message = {"role": "assistant", "content": [{"text": "<think>reasoning ###STOP###</think>Here you go."}]}
        assert extract_message_text(message) == "Here you go."

    def test_unclosed_think_returned_as_is(self):
        message = {"role": "assistant", "content": [{"text": "<think>truncated mid-thought"}]}
        assert extract_message_text(message) == "<think>truncated mid-thought"


# ---------------------------------------------------------------------------
# RolloutResult
# ---------------------------------------------------------------------------


class TestRolloutResult:
    def test_final_response_from_assistant(self):
        messages = [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}, {"text": "world"}]},
        ]
        result = RolloutResult(messages=messages)
        assert result.final_response == "world"

    def test_final_response_strips_think_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "<think>reasoning here</think>The actual answer"}],
            },
        ]
        result = RolloutResult(messages=messages)
        assert result.final_response == "The actual answer"

    def test_final_response_strips_nested_think_tags(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"text": "<think>first</think>middle<think>second</think>final answer"}],
            },
        ]
        result = RolloutResult(messages=messages)
        assert result.final_response == "final answer"

    def test_final_response_no_assistant(self):
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        result = RolloutResult(messages=messages)
        assert result.final_response is None

    def test_final_response_empty(self):
        result = RolloutResult()
        assert result.final_response is None

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
        result = RolloutResult(messages=messages)
        assert result.final_response == "result is 4"

    def test_defaults(self):
        result = RolloutResult()
        assert result.reward_result is None
        assert result.termination_reason == TerminationReason.NOT_TERMINATED

    def test_with_reward(self):
        reward = RewardResult(reward=1.0, info={"exact_match": True})
        result = RolloutResult(reward_result=reward)
        assert result.reward_result.reward == 1.0
        assert result.reward_result.info["exact_match"] is True


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

    def test_max_messages(self):
        error = EventLoopException(Exception())
        error.__cause__ = MaxMessagesReachedError(20)
        assert TerminationReason.from_error(error) == TerminationReason.MAX_MESSAGES_REACHED

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

    def test_recursion_depth_exceeded(self):
        error = EventLoopException(Exception())
        error.__cause__ = RecursionError("maximum recursion depth exceeded")
        assert TerminationReason.from_error(error) == TerminationReason.RECURSION_DEPTH_EXCEEDED

    def test_connection_error_builtin(self):
        error = EventLoopException(Exception())
        error.__cause__ = ConnectionResetError("peer reset")
        assert TerminationReason.from_error(error) == TerminationReason.CONNECTION_ERROR

    def test_server_disconnected_by_name(self):
        class ServerDisconnectedError(Exception):
            pass

        error = EventLoopException(Exception())
        error.__cause__ = ServerDisconnectedError("Server disconnected")
        assert TerminationReason.from_error(error) == TerminationReason.CONNECTION_ERROR

    def test_timeout_takes_precedence_over_connection(self):
        # Both 'timeout' and 'connection' patterns in the chain — timeout wins
        # because it's matched first in `from_error`.
        class ServerTimeoutError(Exception):
            pass

        error = EventLoopException(Exception())
        error.__cause__ = ServerTimeoutError()
        assert TerminationReason.from_error(error) == TerminationReason.TIMEOUT

    def test_generic_error(self):
        error = EventLoopException(Exception())
        error.__cause__ = ValueError("something broke")
        assert TerminationReason.from_error(error) == TerminationReason.UNCLASSIFIED_ERROR

    def test_non_event_loop_exception(self):
        error = RuntimeError("direct error")
        assert TerminationReason.from_error(error) == TerminationReason.UNCLASSIFIED_ERROR
