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

"""User-simulator side of tau2-bench: the LLM user and its dialogue protocol.

`Tau2BenchUserSimulator` owns the user-side agent (persona prompt, user tools) and drives the
whole multi-turn dialogue inside a single `agent.invoke_async()` by feeding each user
reply back via `AfterInvocationEvent.resume`, until a stop marker
(`###STOP###` / `###TRANSFER###` / `###OUT-OF-SCOPE###`) or the `max_steps` budget.

`max_steps` mirrors tau2's step semantics: every message hop (agent message, tool round,
user message) counts one step. It is enforced as a single budget over the total message
count of both agents — a `LoopLimiter` on the user-sim agent is re-armed each turn with
whatever the assistant conversation has not consumed.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Any, ClassVar

from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry
from strands.models import Model
from strands_sglang import LoopLimiter

from strands_env.core.types import TerminationReason, extract_message_text

from . import _tau2
from .tool import Tau2BenchTool

logger = logging.getLogger(__name__)


class Tau2BenchTerminationReason(StrEnum):
    """Why the tau2 dialogue ended, in tau2's own vocabulary.

    Notes:
        Matching tau2's dual mode, only the user can end the dialogue — the standard
        tau2 `LLMAgent` never overrides `is_stop`, so there is no agent-side stop.
    """

    USER_STOP = "user_stop"
    MAX_STEPS = "max_steps"
    ABORTED = "aborted"


class Tau2BenchUserSimulator(HookProvider):
    """Simulated user: owns the user-side agent and drives the dialogue turn-by-turn.

    Registered as a hook on the assistant agent, it replies at each agent loop's end
    via `AfterInvocationEvent.resume`, so the whole dialogue runs inside a single
    `invoke_async()`.
    """

    GREETING_MESSAGE: ClassVar[str] = "Hi! How can I help you today?"
    SYSTEM_PROMPT_TEMPLATE: ClassVar[str] = "{guidelines}\n\n<scenario>\n{scenario}\n</scenario>"
    STOP_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")

    def __init__(
        self,
        *,
        model: Model,
        tools: list[Tau2BenchTool],
        scenario: str,
        max_steps: int = 100,
        verbose: bool = False,
    ):
        """Initialize a `Tau2BenchUserSimulator` instance.

        Args:
            model: Model powering the simulated user.
            scenario: Task-specific user persona and instructions, formatted as a string.
            tools: User-side tools, if the task provides any.
            max_steps: Step budget shared by the assistant conversation and the user
                simulation: the total message count across both agents. A slightly
                conservative approximation of tau2's step count (the seeded greeting and
                each turn's mirrored incoming message consume budget but not tau2 steps).
            verbose: Stream the user-sim's output to stdout.
        """
        self.limiter = LoopLimiter(max_messages=max_steps)
        self.agent = Agent(
            model=model,
            tools=tools,  # type: ignore[arg-type]
            system_prompt=self.SYSTEM_PROMPT_TEMPLATE.format(
                guidelines=_tau2.user_sim_guidelines(use_tools=bool(tools)), scenario=scenario
            ),
            conversation_manager=NullConversationManager(),
            hooks=[self.limiter],
            callback_handler=PrintingCallbackHandler() if verbose else None,
        )
        self.max_steps = max_steps
        self.termination: Tau2BenchTerminationReason = Tau2BenchTerminationReason.ABORTED

    async def first_message(self) -> str:
        """Reply to the canned greeting — seeds the assistant agent's first invoke.

        Mirrors tau2's stop check on every user message: if the reply already carries a
        stop marker (e.g. an out-of-scope scenario), `termination` is set to `USER_STOP`
        and the episode should not reach the assistant.
        """
        reply = await self.agent.invoke_async(self.GREETING_MESSAGE)
        text = extract_message_text(reply.message)
        if self.STOP_PATTERN.search(text):
            self.termination = Tau2BenchTerminationReason.USER_STOP
        return text

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe to `AfterInvocationEvent`."""
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:

        # Extract the agent's response text (no think block)
        agent_text = extract_message_text(event.result.message) if event.result else ""

        # Re-arm the shared step budget: on top of what the user-sim has already consumed,
        # this turn may add at most the steps the assistant conversation has left
        self.limiter.max_messages = self.limiter.message_count + (self.max_steps - len(event.agent.messages))
        try:
            user_result = await self.agent.invoke_async(agent_text)
            user_text = extract_message_text(user_result.message)
        except Exception as e:
            # Check if the error is caused by the max steps limit
            if TerminationReason.from_error(e) is TerminationReason.MAX_MESSAGES_REACHED:
                self.termination = Tau2BenchTerminationReason.MAX_STEPS
                return
            logger.warning("user-sim failed: %s", e)
            self.termination = Tau2BenchTerminationReason.ABORTED
            return

        # Check if the user wanted to stop the dialogue
        if self.STOP_PATTERN.search(user_text):
            self.termination = Tau2BenchTerminationReason.USER_STOP
            # Append the terminating user msg so it shows up in `result.messages`.
            event.agent.messages.append({"role": "user", "content": [{"text": user_text}]})
            return

        # Resume the assistant conversation with simulated user response
        event.resume = user_text
