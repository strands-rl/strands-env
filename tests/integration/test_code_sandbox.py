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

"""Integration tests for CodeSandboxEnv with a real SGLang model.

Requires:
- A running SGLang server (default: http://localhost:30000)
- Valid AWS credentials with Bedrock AgentCore access
"""

import pytest

from strands_env.core.types import Action, RewardResult, StepResult, TaskContext, TerminationReason
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.utils.aws import check_credentials, get_client, get_session

from .conftest import assert_successful_step, assert_token_observation, assert_token_usage

FORCE_TOOL_PROMPT = (
    "You are a coding assistant. Always use the execute_code tool. "
    "Break every problem into many small steps, each requiring a separate code execution."
)

MANY_STEPS_PROMPT = "Compute 1+1, then 2+2, then 3+3, then 4+4, then 5+5, each in a separate code execution."

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def agentcore_client():
    """Create a bedrock-agentcore client, skipping if AWS credentials are not configured."""
    session = get_session()
    if not check_credentials(session):
        pytest.skip("AWS credentials not available")
    return get_client("bedrock-agentcore")


@pytest.fixture
async def code_env(model_factory, agentcore_client):
    """CodeSandboxEnv in CODE mode with automatic cleanup."""
    env = CodeSandboxEnv(model_factory=model_factory, client=agentcore_client, mode="code")
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCodeSandboxEnv:
    async def test_code_mode(self, code_env):
        """CODE mode: agent executes Python, produces complete observation with token trajectory and metrics."""
        result = await code_env.step(Action(message="Use code to compute 2 ** 10 and tell me the result."))

        assert_successful_step(result)
        assert_token_observation(result)
        assert_token_usage(result)

        assert result.observation.metrics["per_tool_metrics"]["execute_code"]["calls"] >= 1
        assert result.observation.metrics["per_tool_metrics"]["execute_code"]["successes"] >= 1

    async def test_terminal_mode(self, model_factory, agentcore_client):
        """TERMINAL mode: agent executes shell commands."""
        env = CodeSandboxEnv(model_factory=model_factory, client=agentcore_client, mode="terminal")
        try:
            result = await env.step(Action(message="Use a shell command to print 'hello' with echo."))
            assert_successful_step(result)
        finally:
            await env.cleanup()

    async def test_code_and_terminal_mode(self, model_factory, agentcore_client):
        """CODE_AND_TERMINAL mode: both execute_code and execute_command tools available."""
        env = CodeSandboxEnv(model_factory=model_factory, client=agentcore_client, mode="code_and_terminal")
        try:
            result = await env.step(
                Action(message="Use code to compute 2 + 2, then use a shell command to echo the result."),
            )
            assert_successful_step(result)
        finally:
            await env.cleanup()

    async def test_multi_turn_with_reward(self, model_factory, agentcore_client):
        """Multi-turn conversation with reward function exercising the full lifecycle."""

        class ContainsReward:
            async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
                response = step_result.observation.final_response or ""
                expected = str(action.task_context.ground_truth)
                return RewardResult(reward=1.0 if expected in response else 0.0)

        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode="code",
            reward_fn=ContainsReward(),
        )
        try:
            result1 = await env.step(Action(message="Use code to define x = 42 and print it."))
            assert result1.termination_reason == TerminationReason.TASK_COMPLETE

            result2 = await env.step(
                Action(
                    message="Now use code to compute x * 2 and give me the final number.",
                    task_context=TaskContext(conversation_history=result1.observation.messages, ground_truth=84),
                ),
            )
            assert result2.termination_reason == TerminationReason.TASK_COMPLETE
            assert result2.reward is not None
            assert isinstance(result2.reward.reward, float)
        finally:
            await env.cleanup()

    async def test_tool_iteration_limit(self, model_factory, agentcore_client):
        """max_tool_iters terminates the agent after the specified number of tool rounds."""
        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode="code",
            system_prompt=FORCE_TOOL_PROMPT,
            max_tool_iters=1,
        )
        try:
            result = await env.step(Action(message=MANY_STEPS_PROMPT))

            assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            assert result.observation.metrics["tool_iters"] <= 1
        finally:
            await env.cleanup()

    async def test_max_tool_calls_limit(self, model_factory, agentcore_client):
        """max_tool_calls terminates the agent after the specified total tool invocations."""
        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode="code",
            system_prompt=FORCE_TOOL_PROMPT,
            max_tool_calls=1,
        )
        try:
            result = await env.step(Action(message=MANY_STEPS_PROMPT))

            assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
            assert result.observation.metrics["tool_calls"] >= 1
        finally:
            await env.cleanup()
