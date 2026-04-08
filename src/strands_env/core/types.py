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

"""Core types for Strands Agents Environments: actions, observations, rewards, model config, and step result."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from strands.types.content import Message, Messages
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException, MaxTokensReachedException
from strands_sglang import MaxToolCallsReachedError, MaxToolIterationsReachedError, TokenManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class TaskContext(BaseModel):
    """Ground truth, conversation history, and arbitrary task-specific fields.

    Extra kwargs are forwarded to reward functions (e.g. `TaskContext(ground_truth="4", difficulty=3)`).
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ground_truth: Any = None
    conversation_history: Messages = Field(default_factory=list)


class Action(BaseModel):
    """A single task: the message to send and the context needed for reward computation."""

    message: str | Message = Field(..., description="The message/prompt to send to the agent.")
    task_context: TaskContext = Field(default_factory=TaskContext)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class TokenObservation(BaseModel):
    """Token-level observation for Token-in/Token-out (TITO) training.

    `prompt_length` splits the flat `token_ids` list into initial-prompt vs. rollout.
    `loss_mask` and `logprobs` cover all tokens (use rollout slices for training).
    """

    token_ids: list[int] = Field(default_factory=list)
    prompt_length: int = Field(default=0)
    loss_mask: list[int] = Field(default_factory=list)
    logprobs: list[float | None] = Field(default_factory=list)

    @property
    def rollout_token_ids(self) -> list[int]:
        """Return token IDs for the rollout (after the initial prompt)."""
        return self.token_ids[self.prompt_length :]

    @property
    def rollout_logprobs(self) -> list[float | None]:
        """Return logprobs for the rollout tokens."""
        return self.logprobs[self.prompt_length :]

    @property
    def rollout_loss_mask(self) -> list[int]:
        """Return loss mask for the rollout tokens."""
        return self.loss_mask[self.prompt_length :]

    @property
    def initial_prompt_token_ids(self) -> list[int]:
        """Return token IDs for the initial prompt."""
        return self.token_ids[: self.prompt_length]

    @classmethod
    def from_token_manager(cls, token_manager: TokenManager) -> TokenObservation | None:
        """Create from strands-sglang's `TokenManager`; returns None if empty."""
        if len(token_manager) == 0:
            return None
        return cls(
            token_ids=token_manager.token_ids,
            prompt_length=len(token_manager.initial_prompt),
            loss_mask=token_manager.loss_mask,
            logprobs=token_manager.logprobs,
        )


class Observation(BaseModel):
    """Step observation: messages produced, optional token data, and metrics."""

    messages: Messages = Field(default_factory=list)
    tokens: TokenObservation | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    routed_experts: str | None = None

    @property
    def final_response(self) -> str | None:
        """Return text from the last assistant message, or None."""
        return self.get_final_response(self.messages)

    @staticmethod
    def get_final_response(messages: Messages) -> str | None:
        """Extract text from the last assistant message, or None."""
        if not messages or messages[-1].get("role") != "assistant":
            return None
        content = messages[-1].get("content", [])
        # Take the last text block — the final textual output.
        text = next(
            (block["text"] for block in reversed(content) if isinstance(block, dict) and "text" in block),
            None,
        )
        if text is None:
            return None
        # Strip think block if any
        think_end = text.rfind("</think>") # find the last </think> tag
        if think_end != -1:
            text = text[think_end + len("</think>") :].lstrip()
        return text or None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class RewardResult(BaseModel):
    """Scalar reward plus optional diagnostics."""

    reward: float = Field(...)
    info: dict[str, Any] = Field(default_factory=dict)


class RewardFunction(ABC):
    """Abstract reward function. Subclass and implement `compute`."""

    @abstractmethod
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Return a `RewardResult` given the action and the environment's step result."""
        ...


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


class TerminationReason(str, Enum):
    """Why an episode ended."""

    NOT_TERMINATED = "not_terminated"
    TASK_COMPLETE = "task_complete"
    MAX_TOKENS_REACHED = "max_tokens_reached"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    MAX_TOOL_ITERATIONS_REACHED = "max_tool_iterations_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    TIMEOUT = "timeout"
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
    def from_error(cls, error: Exception | None) -> TerminationReason:
        """Map an agent exception to a `TerminationReason`.

        Unwraps `EventLoopException` to inspect the underlying cause.
        """
        if error is None:
            return cls.TASK_COMPLETE

        cause = error.__cause__ if isinstance(error, EventLoopException) else error

        match cause:
            case MaxTokensReachedException():
                reason = cls.MAX_TOKENS_REACHED
            case ContextWindowOverflowException():
                reason = cls.CONTEXT_WINDOW_OVERFLOW
            case MaxToolIterationsReachedError():
                reason = cls.MAX_TOOL_ITERATIONS_REACHED
            case MaxToolCallsReachedError():
                reason = cls.MAX_TOOL_CALLS_REACHED
            case e if cls._is_timeout(e):
                reason = cls.TIMEOUT
            case _:
                reason = cls.UNCLASSIFIED_ERROR

        logger.warning("Step terminated: %s - %s", reason.value, cause)
        return reason


class StepResult(BaseModel):
    """Result of a single `Environment.step` call."""

    observation: Observation
    reward: RewardResult | None = None
    termination_reason: TerminationReason = TerminationReason.NOT_TERMINATED
