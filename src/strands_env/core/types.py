from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
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
    trace_attributes: dict[str, str] | None = Field(
        default=None, description="Per-sample OTel trace attributes for the rollout span."
    )


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


class TerminationReason(StrEnum):
    """Why an episode ended."""

    NOT_TERMINATED = "not_terminated"
    TASK_COMPLETE = "task_complete"

    # Loop budget exhaustion.
    MAX_TOKENS_REACHED = "max_tokens_reached"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    MAX_TOOL_ITERATIONS_REACHED = "max_tool_iterations_reached"
    MAX_TOOL_CALLS_REACHED = "max_tool_calls_reached"
    MAX_MESSAGES_REACHED = "max_messages_reached"
    RECURSION_DEPTH_EXCEEDED = "recursion_depth_exceeded"

    # Keyword-based reasons. Declaration order is match priority — see `keywords`.
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    AUTH_ERROR = "auth_error"
    THROTTLED = "throttled"

    # All other errors that haven't been classified yet.
    UNCLASSIFIED_ERROR = "unclassified_error"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Lowercase keywords that classify an exception as this reason, matched by `from_keywords`.

        Empty for reasons recognized by exception type instead. Keeping the table here rather
        than in a module constant keeps it beside the members it maps.
        """
        return {
            TerminationReason.TIMEOUT: ("timeout",),
            TerminationReason.CONNECTION_ERROR: ("connection", "disconnected"),
            TerminationReason.AUTH_ERROR: (
                "expiredtoken",
                "credential",
                "accessdenied",
                "unauthoriz",
                "authentication",
                "unrecognizedclient",
                "permissiondenied",
            ),
            # Reaching us means Strands' event loop already exhausted its retries.
            # Not "limitexceeded": that also matches resource caps like `ItemCollectionSizeLimit`.
            TerminationReason.THROTTLED: ("throttl", "ratelimit", "toomanyrequests"),
        }.get(self, ())

    @classmethod
    def from_keywords(cls, error: BaseException | None) -> TerminationReason | None:
        """Classify an exception by `keywords`, or None if nothing matches.

        Matches the exception class name, plus the AWS error code for boto errors — those share
        one uninformative `ClientError` name. Reasons are tried in declaration order, each against
        the whole `__cause__` chain, so `timeout` beats `connection` for a `ConnectTimeoutError`.
        """
        for reason in cls:
            if not reason.keywords:
                continue
            exc = error
            while exc is not None:
                response = getattr(exc, "response", None)
                code = response.get("Error", {}).get("Code", "") if isinstance(response, dict) else ""
                identity = f"{type(exc).__name__} {code}".lower()
                if any(keyword in identity for keyword in reason.keywords):
                    return reason
                exc = exc.__cause__
        return None

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
            case _:
                reason = cls.from_keywords(cause) or cls.UNCLASSIFIED_ERROR

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
        """Return text from the last assistant message, with think blocks stripped."""
        return self.get_final_response(raw=False)

    def get_final_response(self, *, raw: bool = False) -> str | None:
        """Return text from the agent's last reply, or None if it did not have the last word.

        Args:
            raw: If True, keep `<think>...</think>` blocks instead of stripping them.
        """
        for message in reversed(self.messages):
            # Blank trailing turns: a model can report `tool_use` with no tool-use block, and the
            # agent loop then appends an empty message and recurses past a completed answer.
            if not extract_message_text(message, raw=True).strip():
                continue
            return extract_message_text(message, raw=raw) or None if message.get("role") == "assistant" else None
        return None


def extract_message_text(message: Message, *, raw: bool = False) -> str:
    """Extract the final text from a message: the last text block.

    Args:
        message: A strands Message dict.
        raw: If True, return the full text including `<think>...</think>` blocks.
            If False (default), strip think blocks before returning.

    Returns an empty string when the message contains no text block.
    An unclosed `<think>` block (truncated generation) is returned as-is.
    """
    content = message.get("content") or []
    text = next(
        (block["text"] for block in reversed(content) if isinstance(block, dict) and "text" in block),
        None,
    )
    if text is None:
        return ""
    if not raw:
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>") :].lstrip()
    return text
