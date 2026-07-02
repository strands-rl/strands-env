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

"""Unit tests for Tau2BenchUserSimulator.

Patches the `Agent` class inside the simulator module so no model or server is needed,
then feeds real `AfterInvocationEvent` objects to exercise the actual hook path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from strands.hooks import AfterInvocationEvent
from strands.types.exceptions import EventLoopException
from strands_sglang import MaxMessagesReachedError

from strands_env.environments.tau2_bench.simulator import (
    Tau2BenchTerminationReason,
    Tau2BenchUserSimulator,
)

MAX_STEPS = 20


@pytest.fixture
def agent_cls():
    """Patch the Agent class in the simulator module for the whole test."""
    with patch("strands_env.environments.tau2_bench.simulator.Agent") as cls:
        cls.return_value = MagicMock(invoke_async=AsyncMock())
        yield cls


@pytest.fixture
def tau2_shim():
    """Patch the _tau2 shim so guidelines derivation needs no tau2 install or data dir."""
    with patch("strands_env.environments.tau2_bench.simulator._tau2") as shim:
        shim.user_sim_guidelines.return_value = "Play the customer."
        yield shim


@pytest.fixture
def sim(agent_cls, tau2_shim):
    return Tau2BenchUserSimulator(
        model=MagicMock(),
        tools=[],
        scenario="Book a flight to Tokyo.",
        max_steps=MAX_STEPS,
    )


def _agent_result(text: str) -> MagicMock:
    """A stand-in for AgentResult exposing only `.message`."""
    return MagicMock(message={"role": "assistant", "content": [{"text": text}]})


def _event(agent_text: str | None, main_message_count: int = 4) -> AfterInvocationEvent:
    """Build a real AfterInvocationEvent carrying a mock main agent with N messages."""
    main_agent = MagicMock()
    main_agent.messages = [{"role": "user", "content": [{"text": f"m{i}"}]} for i in range(main_message_count)]
    result = _agent_result(agent_text) if agent_text is not None else None
    return AfterInvocationEvent(agent=main_agent, result=result)


# =============================================================================
# Construction
# =============================================================================


class TestInit:
    def test_system_prompt_combines_guidelines_and_scenario(self, agent_cls, sim):
        system_prompt = agent_cls.call_args.kwargs["system_prompt"]
        assert system_prompt == "Play the customer.\n\n<scenario>\nBook a flight to Tokyo.\n</scenario>"

    def test_limiter_attached_with_full_budget(self, agent_cls, sim):
        assert agent_cls.call_args.kwargs["hooks"] == [sim.limiter]
        assert sim.limiter.max_messages == MAX_STEPS

    def test_termination_starts_aborted(self, sim):
        assert sim.termination is Tau2BenchTerminationReason.ABORTED

    @pytest.mark.parametrize(("tools", "use_tools"), [([], False), ([MagicMock()], True)])
    def test_guidelines_derived_like_tau2(self, agent_cls, tau2_shim, tools, use_tools):
        """Guidelines always come from tau2's global guidelines: tools variant iff the user has tools."""
        Tau2BenchUserSimulator(model=MagicMock(), tools=tools, scenario="S.", max_steps=MAX_STEPS)
        tau2_shim.user_sim_guidelines.assert_called_once_with(use_tools=use_tools)
        assert agent_cls.call_args.kwargs["system_prompt"].startswith("Play the customer.")


# =============================================================================
# first_message
# =============================================================================


class TestFirstMessage:
    async def test_replies_to_canned_greeting(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("Hi, I need a flight to Tokyo.")
        text = await sim.first_message()
        assert text == "Hi, I need a flight to Tokyo."
        sim.agent.invoke_async.assert_awaited_once_with(Tau2BenchUserSimulator.GREETING_MESSAGE)
        assert sim.termination is Tau2BenchTerminationReason.ABORTED

    async def test_stop_marker_sets_user_stop(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("###OUT-OF-SCOPE###")
        await sim.first_message()
        assert sim.termination is Tau2BenchTerminationReason.USER_STOP

    async def test_marker_inside_think_block_is_not_a_stop(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("<think>should I say ###STOP###?</think>Hello!")
        text = await sim.first_message()
        assert text == "Hello!"
        assert sim.termination is Tau2BenchTerminationReason.ABORTED


# =============================================================================
# _on_after_invocation
# =============================================================================


class TestOnAfterInvocation:
    async def test_normal_turn_resumes_with_user_reply(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("Economy, please.")
        event = _event("Which class would you like?")
        await sim._on_after_invocation(event)
        sim.agent.invoke_async.assert_awaited_once_with("Which class would you like?")
        assert event.resume == "Economy, please."
        assert sim.termination is Tau2BenchTerminationReason.ABORTED

    async def test_agent_think_block_stripped_before_relay(self, sim):
        """The user-sim must not see the agent's private reasoning."""
        sim.agent.invoke_async.return_value = _agent_result("Sure.")
        await sim._on_after_invocation(_event("<think>check policy first</think>What date?"))
        sim.agent.invoke_async.assert_awaited_once_with("What date?")

    async def test_none_result_relays_empty_text(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("Hello? Anyone there?")
        await sim._on_after_invocation(_event(None))
        sim.agent.invoke_async.assert_awaited_once_with("")

    @pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"])
    async def test_user_stop_markers_terminate_and_record_message(self, sim, marker):
        sim.agent.invoke_async.return_value = _agent_result(f"Thanks, that's all. {marker}")
        event = _event("Anything else?")
        await sim._on_after_invocation(event)
        assert sim.termination is Tau2BenchTerminationReason.USER_STOP
        assert event.resume is None
        # The terminating user message is appended so it shows up in result.messages
        assert event.agent.messages[-1] == {
            "role": "user",
            "content": [{"text": f"Thanks, that's all. {marker}"}],
        }

    async def test_user_marker_inside_think_block_is_not_a_stop(self, sim):
        sim.agent.invoke_async.return_value = _agent_result("<think>###STOP###?</think>One more thing.")
        event = _event("Anything else?")
        await sim._on_after_invocation(event)
        assert event.resume == "One more thing."
        assert sim.termination is Tau2BenchTerminationReason.ABORTED

    async def test_budget_rearmed_from_main_conversation(self, sim):
        """This turn may add at most what the assistant conversation has not consumed."""
        sim.agent.invoke_async.return_value = _agent_result("Okay.")
        sim.limiter.message_count = 3  # user-sim already consumed 3
        await sim._on_after_invocation(_event("Next?", main_message_count=6))
        assert sim.limiter.max_messages == 3 + (MAX_STEPS - 6)

    async def test_budget_overrun_classified_as_max_steps(self, sim):
        error = EventLoopException(Exception())
        error.__cause__ = MaxMessagesReachedError(MAX_STEPS)
        sim.agent.invoke_async.side_effect = error
        event = _event("Anything else?")
        await sim._on_after_invocation(event)
        assert sim.termination is Tau2BenchTerminationReason.MAX_STEPS
        assert event.resume is None

    async def test_other_error_aborts(self, sim):
        sim.agent.invoke_async.side_effect = ValueError("boom")
        event = _event("Anything else?")
        await sim._on_after_invocation(event)
        assert sim.termination is Tau2BenchTerminationReason.ABORTED
        assert event.resume is None

    async def test_same_agent_across_turns(self, agent_cls, sim):
        """Regression: the user-sim keeps one agent (and its dialogue memory) for the episode."""
        sim.agent.invoke_async.return_value = _agent_result("Reply.")
        await sim._on_after_invocation(_event("Turn one?"))
        await sim._on_after_invocation(_event("Turn two?"))
        assert agent_cls.call_count == 1  # constructed once in __init__, never per turn
        assert sim.agent.invoke_async.await_count == 2
