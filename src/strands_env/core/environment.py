# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Environment class for `strands-env`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from strands import Agent
from strands.agent.conversation_manager import ConversationManager, NullConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.telemetry.metrics import EventLoopMetrics
from strands_sglang import TokenManager, ToolLimiter

from .models import ModelFactory
from .types import (
    Action,
    Observation,
    RewardFunction,
    StepResult,
    TerminationReason,
    TokenObservation,
)

logger = logging.getLogger(__name__)


class Environment:
    """Base RL rollout environment for Strands agents."""

    default_system_prompt_path: ClassVar[Path | None] = None

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        system_prompt: str | None = None,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        max_parallel_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        """Initialize the environment with model factory, prompts, and limits."""
        self.model_factory = model_factory
        self.reward_fn = reward_fn
        self.max_tool_iters = max_tool_iters
        self.max_tool_calls = max_tool_calls
        self.max_parallel_tool_calls = max_parallel_tool_calls
        self.verbose = verbose

        path = self.default_system_prompt_path
        self.system_prompt = system_prompt or (path.read_text() if path and path.exists() else None)

    async def reset(self) -> None:
        """Reset for a new episode. Override for environment-specific init.

        This is the right place for resource-heavy or async initialization
        (e.g., spinning up containers, creating sessions, connecting to services).
        Keep `__init__` limited to storing config and lightweight state —
        it is synchronous and cannot `await`.

        Paired with `cleanup` which tears down what `reset` sets up.
        """
        pass

    async def step(self, action: Action) -> StepResult:
        """Run one agent episode and return observation + reward + termination."""
        conversation_history = action.task_context.conversation_history
        tool_limiter = ToolLimiter(
            max_tool_iters=self.max_tool_iters,
            max_tool_calls=self.max_tool_calls,
            max_parallel_tool_calls=self.max_parallel_tool_calls,
        )
        model = self.model_factory()
        agent = Agent(
            model=model,
            messages=list(conversation_history),
            tools=list(self.get_tools()),
            system_prompt=self.system_prompt,
            hooks=[tool_limiter] + list(self.get_hooks()),
            conversation_manager=self.get_conversation_manager(),
            callback_handler=PrintingCallbackHandler() if self.verbose else None,
        )
        error = None
        try:
            message = action.message if isinstance(action.message, str) else action.message["content"]
            await agent.invoke_async(message)
        except Exception as e:
            error = e
        termination_reason = TerminationReason.from_error(error)

        step_messages = list(agent.messages)[len(conversation_history) :]
        token_obs = TokenObservation.from_token_manager(getattr(agent.model, "token_manager", TokenManager()))
        tool_parse_errors = getattr(agent.model, "tool_parse_errors", None)
        metrics = {
            "message_count": len(step_messages),
            "tool_iters": tool_limiter.tool_iter_count,
            "tool_calls": tool_limiter.tool_call_count,
            "cancelled_tool_calls": tool_limiter.cancelled_tool_call_count,
            **self.compute_metrics(agent.event_loop_metrics, tool_parse_errors=tool_parse_errors),
        }
        observation = Observation(messages=step_messages, tokens=token_obs, metrics=metrics)
        step_result = StepResult(observation=observation, termination_reason=termination_reason)
        step_result.reward = (
            (await self.reward_fn.compute(action=action, step_result=step_result)) if self.reward_fn else None
        )
        return step_result

    async def cleanup(self) -> None:
        """Release resources. Override in subclasses."""
        pass

    def get_tools(self) -> list:
        """Tools available to the agent. Override in subclasses."""
        return []

    def get_hooks(self) -> list:
        """Agent hooks. Override and call `super()` to extend."""
        return []

    def get_conversation_manager(self) -> ConversationManager:
        """Conversation manager for context window handling. Override in subclasses."""
        return NullConversationManager()

    def compute_metrics(
        self,
        event_loop_metrics: EventLoopMetrics,
        tool_parse_errors: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Extract metrics from the event loop. Override to add custom metrics."""

        def _summarize(counts: Sequence[Any], round_digits: int = 1) -> dict[str, int | float]:
            return {
                "total": round(sum(counts), round_digits),
                "max": round(max(counts), round_digits),
                "mean": round(sum(counts) / len(counts), round_digits),
                "min": round(min(counts), round_digits),
            }

        per_model_call_usage = [
            (cycle.usage.get("inputTokens", 0), cycle.usage.get("outputTokens", 0))
            for invocation in event_loop_metrics.agent_invocations
            for cycle in invocation.cycles
        ]
        input_counts, output_counts = zip(*per_model_call_usage, strict=True) if per_model_call_usage else ([], [])
        cycle_durations = event_loop_metrics.cycle_durations

        per_tool_metrics = {
            name: {
                "calls": tm.call_count,
                "successes": tm.success_count,
                "errors": tm.error_count,
                "parse_errors": (tool_parse_errors or {}).get(name, 0),
                "latency_s": round(tm.total_time, 4),
            }
            for name, tm in event_loop_metrics.tool_metrics.items()
        }

        return {
            "model_calls": event_loop_metrics.cycle_count,
            "model_latency_s": _summarize(cycle_durations, round_digits=4) if cycle_durations else None,
            "input_tokens": _summarize(input_counts) if input_counts else None,
            "output_tokens": _summarize(output_counts) if output_counts else None,
            "per_tool_metrics": per_tool_metrics or None,
        }
