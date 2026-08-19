"""Integration tests for MathEnv.

Exercises the full rollout lifecycle: agent invocation → result
(messages, tokens, metrics) → optional reward — against a real SGLang model.

Requires a running SGLang server (default: http://localhost:30000).
"""

from strands_env.core.types import Task, TerminationReason
from strands_env.environments.math import MathEnv
from strands_env.environments.math.reward import MathVerifyReward

from .conftest import assert_rollout, assert_successful_rollout, assert_token_usage

MATH_SYSTEM_PROMPT = "You are a math assistant. Solve problems step by step. Be concise."


class TestMathEnv:
    async def test_rollout_produces_complete_result(self, model_factory):
        """A single rollout produces a complete result with messages, token trajectory, and metrics."""
        env = MathEnv(model_factory=model_factory, system_prompt=MATH_SYSTEM_PROMPT)
        result = await env.rollout(Task(message="What is 17 * 23?"))

        assert_successful_rollout(result)
        assert_rollout(result)
        assert_token_usage(result)

        # No tools in this env, so the agent answers in one model call and never loops.
        assert result.metrics["tool_calls"] == 0
        assert result.metrics["tool_iters"] == 0

    async def test_multi_turn_conversation(self, model_factory):
        """Agent uses conversation history from a prior turn to maintain context."""
        env = MathEnv(model_factory=model_factory, system_prompt=MATH_SYSTEM_PROMPT)

        result1 = await env.rollout(Task(message="What is 10 + 5?"))
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        result2 = await env.rollout(
            Task(
                message="Now multiply that result by 3.",
                conversation_history=result1.messages,
            )
        )
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_reward_fn(self, model_factory):
        """MathVerifyReward computes a symbolic-match reward from the agent's boxed answer."""
        env = MathEnv(
            model_factory=model_factory,
            system_prompt=MATH_SYSTEM_PROMPT + " Put your final answer inside \\boxed{}.",
            reward_fn=MathVerifyReward(),
        )
        result = await env.rollout(
            Task(message="What is 6 * 7?", ground_truth="42"),
        )

        assert result.reward_result is not None
        assert isinstance(result.reward_result.reward, float)
        assert "matched" in result.reward_result.info or "reason" in result.reward_result.info
