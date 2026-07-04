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

"""Core types for Strands Agents Environments: tasks, rewards, model config, and rollout results."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Generic

from pydantic import BaseModel, ConfigDict, Field
from strands.types.content import Message, Messages
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException, MaxTokensReachedException
from strands_sglang import MaxMessagesReachedError, MaxToolCallsReachedError, MaxToolIterationsReachedError, Rollout
from typing_extensions import TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task (input to a rollout)
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A single task containing the starting message and any additional context.

    Environments with per-sample payload should subclass `Task` with declared, typed fields
    (e.g. a serialized task spec, per-task directories) — declared fields are validated as
    usual. `extra="allow"` keeps undeclared fields for ad-hoc tasks without a subclass.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the task.")
    message: str | Message = Field(..., description="The task message/prompt to send to the agent.")
    ground_truth: Any = Field(default=None, description="The ground truth answer to the task.")
    conversation_history: Messages = Field(default_factory=list, description="The conversation prior to the task.")


#: The task type an environment (or reward function) consumes. Defaults to `Task` (PEP 696),
TaskT = TypeVar("TaskT", bound=Task, default=Task)


# ---------------------------------------------------------------------------
# Reward (for a rollout)
# ---------------------------------------------------------------------------


class RewardResult(BaseModel):
    """Scalar reward plus optional diagnostic information."""

    reward: float = Field(..., description="The reward scalar value.")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional diagnostic information.")


class RewardFunction(ABC, Generic[TaskT]):
    """Abstract reward function. Subclass and implement `compute`."""

    @abstractmethod
    async def compute(self, task: TaskT, result: RolloutResult) -> RewardResult:
        """Return a `RewardResult` given the task and the rollout result."""
        ...


# ---------------------------------------------------------------------------
# Termination reason (why a rollout ended)
# ---------------------------------------------------------------------------


class TerminationReason(str, Enum):
    """Why an episode ended."""

    NOT_TERMINATED = "not_terminated"
    TASK_COMPLETE = "task_complete"
    MAX_TOKENS_REACHED = "max_tokens_reached"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    MAX_TOOL_ITERATIONS_REACHED = "max_tool_iterations_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    MAX_MESSAGES_REACHED = "max_messages_reached"
    RECURSION_DEPTH_EXCEEDED = "recursion_depth_exceeded"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    UNCLASSIFIED_ERROR = "unclassified_error"

    @classmethod
    def _is_timeout(cls, error: BaseException | None) -> bool:
        """Check if any exception in the cause chain is a timeout (backend-agnostic)."""
        exc = error
        while exc is not None:
            if "timeout" in type(exc).__name__.lower():
                return True
            exc = exc.__cause__
        return False

    @classmethod
    def _is_connection_error(cls, error: BaseException | None) -> bool:
        """Check if any exception in the cause chain is a connection-level failure."""
        exc = error
        while exc is not None:
            name = type(exc).__name__.lower()
            if "connection" in name or "disconnected" in name:
                return True
            exc = exc.__cause__
        return False

    @classmethod
    def from_error(cls, error: Exception | None) -> TerminationReason:
        """Map an agent exception to a `TerminationReason`.

        Walks the `__cause__` chain past nested `EventLoopException`s — Strands
        re-raises a fresh `EventLoopException` at every recursive `event_loop_cycle`,
        so deep tool-call paths produce multi-level wrappings.
        """
        if error is None:
            return cls.TASK_COMPLETE

        cause: BaseException | None = error
        while isinstance(cause, EventLoopException) and cause.__cause__ is not None:
            cause = cause.__cause__

        match cause:
            case MaxTokensReachedException():
                reason = cls.MAX_TOKENS_REACHED
            case ContextWindowOverflowException():
                reason = cls.CONTEXT_WINDOW_OVERFLOW
            case MaxToolIterationsReachedError():
                reason = cls.MAX_TOOL_ITERATIONS_REACHED
            case MaxToolCallsReachedError():
                reason = cls.MAX_TOOL_CALLS_REACHED
            case MaxMessagesReachedError():
                reason = cls.MAX_MESSAGES_REACHED
            case RecursionError():
                reason = cls.RECURSION_DEPTH_EXCEEDED
            case e if cls._is_timeout(e):
                reason = cls.TIMEOUT
            case e if cls._is_connection_error(e):
                reason = cls.CONNECTION_ERROR
            case _:
                reason = cls.UNCLASSIFIED_ERROR

        logger.warning("Rollout terminated: %s - %s", reason.value, cause)
        return reason


# ---------------------------------------------------------------------------
# Rollout result (output of a rollout)
# ---------------------------------------------------------------------------


class RolloutResult(BaseModel):
    """Result of a single `Environment.rollout` call: trajectory, reward, and termination."""

    messages: Messages = Field(default_factory=list)
    rollout: Rollout | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    reward_result: RewardResult | None = None
    termination_reason: TerminationReason = TerminationReason.NOT_TERMINATED

    @property
    def final_response(self) -> str | None:
        """Return text from the last assistant message, or None.

        None when the conversation ended on a non-assistant message (e.g. a tau2
        `user_stop` turn) — the agent did not have the last word.
        """
        if not self.messages or self.messages[-1].get("role") != "assistant":
            return None
        return extract_message_text(self.messages[-1]) or None


def extract_message_text(message: Message) -> str:
    """Extract the final text from a message: the last text block, with any think block stripped.

    Returns an empty string when the message contains no text block.
    An unclosed `<think>` block (truncated generation) is returned as-is.
    """
    content = message.get("content") or []
    # Take the last text block — the final textual output.
    text = next(
        (block["text"] for block in reversed(content) if isinstance(block, dict) and "text" in block),
        None,
    )
    if text is None:
        return ""
    # Strip think block if any: keep only what follows the last closing tag
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].lstrip()
    return text
