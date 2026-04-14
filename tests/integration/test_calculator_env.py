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

"""Integration tests for CalculatorEnv.

Exercises the full step lifecycle: agent invocation → observation
(messages, tokens, metrics) → optional reward — against a real SGLang model.

Requires a running SGLang server (default: http://localhost:30000).
"""

from strands_env.core.types import Action, TaskContext, TerminationReason
from strands_env.environments.calculator import CalculatorEnv
from strands_env.rewards.math_verify_reward import MathVerifyReward

from .conftest import assert_successful_step, assert_token_observation, assert_token_usage

MATH_SYSTEM_PROMPT = "You are a math assistant. Use the calculator tool to solve problems. Be concise."

FORCE_TOOL_PROMPT = (
    "You are a math assistant. Use the calculator tool for every single step. "
    "Break every problem into many small steps, each requiring a separate calculator call."
)

MANY_STEPS_PROMPT = "Compute 1+1, then 2+2, then 3+3, then 4+4, then 5+5 one at a time."


class TestCalculatorEnv:
    async def test_step_produces_complete_observation(self, model_factory):
        """A single step produces a complete observation with messages, token trajectory, and metrics."""
        env = CalculatorEnv(model_factory=model_factory, system_prompt=MATH_SYSTEM_PROMPT)
        result = await env.step(Action(message="What is 17 * 23?"))

        assert_successful_step(result)
        assert_token_observation(result)
        assert_token_usage(result)

        # Per-tool breakdown for calculator
        per_tool = result.observation.metrics["per_tool_metrics"]
        assert per_tool is not None
        assert per_tool["calculator"]["calls"] >= 1
        assert per_tool["calculator"]["successes"] >= 1
        assert "latency_s" in per_tool["calculator"]

    async def test_multi_turn_conversation(self, model_factory):
        """Agent uses conversation history from a prior turn to maintain context."""
        env = CalculatorEnv(model_factory=model_factory, system_prompt=MATH_SYSTEM_PROMPT)

        result1 = await env.step(Action(message="What is 10 + 5?"))
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        result2 = await env.step(
            Action(
                message="Now multiply that result by 3.",
                task_context=TaskContext(conversation_history=result1.observation.messages),
            )
        )
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_reward_fn(self, model_factory):
        """MathVerifyReward computes a symbolic-match reward from the agent's boxed answer."""
        env = CalculatorEnv(
            model_factory=model_factory,
            system_prompt=MATH_SYSTEM_PROMPT + " Put your final answer inside \\boxed{}.",
            reward_fn=MathVerifyReward(),
        )
        result = await env.step(
            Action(message="What is 6 * 7?", task_context=TaskContext(ground_truth="42")),
        )

        assert result.reward is not None
        assert isinstance(result.reward.reward, float)
        assert "matched" in result.reward.info or "reason" in result.reward.info

    async def test_tool_iteration_limit(self, model_factory):
        """max_tool_iters terminates the agent after the specified number of tool rounds."""
        env = CalculatorEnv(model_factory=model_factory, system_prompt=FORCE_TOOL_PROMPT, max_tool_iters=1)
        result = await env.step(Action(message=MANY_STEPS_PROMPT))

        assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
        assert result.observation.metrics["tool_iters"] <= 1

    async def test_max_tool_calls_limit(self, model_factory):
        """max_tool_calls terminates the agent after the specified total tool invocations."""
        env = CalculatorEnv(model_factory=model_factory, system_prompt=FORCE_TOOL_PROMPT, max_tool_calls=1)
        result = await env.step(Action(message=MANY_STEPS_PROMPT))

        assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
        assert result.observation.metrics["tool_calls"] >= 1
