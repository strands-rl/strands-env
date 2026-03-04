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

"""Integration tests for CodeSandboxEnv with a real SGLang model.

Requires:
- A running SGLang server (default: http://localhost:30000)
- Valid AWS credentials with Bedrock AgentCore access
"""

import pytest

from strands_env.core.types import Action, RewardResult, StepResult, TaskContext, TerminationReason
from strands_env.environments.code_sandbox import CodeMode, CodeSandboxEnv
from strands_env.utils.aws import check_credentials, get_client, get_session

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
    """CodeSandboxEnv in CODE mode with cleanup."""
    env = CodeSandboxEnv(
        model_factory=model_factory,
        client=agentcore_client,
        mode=CodeMode.CODE,
    )
    yield env
    await env.cleanup()


@pytest.fixture
async def terminal_env(model_factory, agentcore_client):
    """CodeSandboxEnv in TERMINAL mode with cleanup."""
    env = CodeSandboxEnv(
        model_factory=model_factory,
        client=agentcore_client,
        mode=CodeMode.TERMINAL,
    )
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# Tests — CODE mode
# ---------------------------------------------------------------------------


class TestCodeMode:
    async def test_step_completes(self, code_env):
        """Agent can execute Python code and complete normally."""
        action = Action(message="What is 2 ** 10? Use code to compute it.")
        result = await code_env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.messages
        assert result.observation.metrics["message_count"] > 0

    async def test_step_produces_token_observation(self, code_env):
        """SGLang model produces token-level observations for TITO training."""
        action = Action(message="Use code to print 'hello world'.")
        result = await code_env.step(action)

        tokens = result.observation.tokens
        assert tokens is not None
        assert len(tokens.token_ids) > 0
        assert tokens.prompt_length > 0
        assert len(tokens.rollout_token_ids) > 0
        # Structural invariants: loss_mask and logprobs must align with token_ids
        assert len(tokens.loss_mask) == len(tokens.token_ids)
        assert len(tokens.logprobs) == len(tokens.token_ids)
        # Rollout logprobs should contain actual float values (not all None)
        rollout_lp = tokens.rollout_logprobs
        assert any(lp is not None for lp in rollout_lp)

    async def test_step_metrics(self, code_env):
        """Step produces expected metric keys with correct structure."""
        action = Action(message="Use code to compute the sum of [1, 2, 3, 4, 5].")
        result = await code_env.step(action)

        metrics = result.observation.metrics
        assert "message_count" in metrics
        assert "tool_iters" in metrics
        assert "tool_calls" in metrics
        assert "model_calls" in metrics
        assert metrics["model_calls"] >= 1
        # Token usage dicts have total/max/mean/min
        for key in ("input_tokens", "output_tokens"):
            usage = metrics[key]
            assert isinstance(usage, dict)
            for subkey in ("total", "max", "mean", "min"):
                assert subkey in usage
                assert usage[subkey] > 0
        # Per-tool metrics: execute_code should appear with correct structure
        per_tool = metrics.get("per_tool_metrics")
        assert per_tool is not None
        assert "execute_code" in per_tool
        tool_m = per_tool["execute_code"]
        assert tool_m["calls"] >= 1
        assert tool_m["successes"] >= 1
        assert "latency_s" in tool_m

    async def test_final_response(self, code_env):
        """Observation provides final assistant response text."""
        action = Action(message="Use code to compute 7 * 8 and tell me the result.")
        result = await code_env.step(action)

        response = result.observation.final_response
        assert response is not None
        assert len(response) > 0

    async def test_conversation_history(self, code_env):
        """Multi-turn interaction with conversation history."""
        action1 = Action(message="Use code to define x = 42 and print it.")
        result1 = await code_env.step(action1)
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        all_messages = result1.observation.messages
        action2 = Action(
            message="Now use code to compute x * 2 and print the result.",
            task_context=TaskContext(conversation_history=all_messages),
        )
        result2 = await code_env.step(action2)
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_reward_fn_called(self, model_factory, agentcore_client):
        """Reward function is invoked and result attached to StepResult."""

        class ContainsReward:
            async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
                response = step_result.observation.final_response or ""
                expected = str(action.task_context.ground_truth)
                match = expected in response
                return RewardResult(reward=1.0 if match else 0.0, info={"match": match})

        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode=CodeMode.CODE,
            reward_fn=ContainsReward(),
        )
        try:
            action = Action(
                message="Use code to compute 7 * 8 and give me the final number.",
                task_context=TaskContext(ground_truth=56),
            )
            result = await env.step(action)

            assert result.reward is not None
            assert isinstance(result.reward.reward, float)
            assert "match" in result.reward.info
        finally:
            await env.cleanup()


# ---------------------------------------------------------------------------
# Tests — TERMINAL mode
# ---------------------------------------------------------------------------


class TestTerminalMode:
    async def test_step_completes(self, terminal_env):
        """Agent can execute shell commands and complete normally."""
        action = Action(message="Use a shell command to print 'hello' with echo.")
        result = await terminal_env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.messages
        assert result.observation.metrics["message_count"] > 0


# ---------------------------------------------------------------------------
# Tests — CODE_AND_TERMINAL mode
# ---------------------------------------------------------------------------


class TestCodeAndTerminalMode:
    async def test_step_completes(self, model_factory, agentcore_client):
        """Agent has both code and terminal tools available."""
        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode=CodeMode.CODE_AND_TERMINAL,
        )
        try:
            action = Action(message="Use code to compute 2 + 2, then use a shell command to echo the result.")
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.TASK_COMPLETE
            assert result.observation.messages
        finally:
            await env.cleanup()


# ---------------------------------------------------------------------------
# Tests — tool limits
# ---------------------------------------------------------------------------


class TestToolLimit:
    async def test_tool_iteration_limit(self, model_factory, agentcore_client):
        """Environment respects max_tool_iters."""
        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode=CodeMode.CODE,
            system_prompt=(
                "You are a coding assistant. Always use the execute_code tool. "
                "Break every problem into many small steps, each requiring a separate code execution."
            ),
            max_tool_iters=1,
        )
        try:
            action = Action(
                message="Compute 1+1, then 2+2, then 3+3, then 4+4, then 5+5, each in a separate code execution."
            )
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            assert result.observation.metrics["tool_iters"] <= 1
        finally:
            await env.cleanup()

    async def test_max_tool_calls(self, model_factory, agentcore_client):
        """Environment respects max_tool_calls (distinct from max_tool_iters).

        Note: parallel tool calls within a single iteration may exceed the limit
        before the limiter fires, so we only assert on termination reason.
        """
        env = CodeSandboxEnv(
            model_factory=model_factory,
            client=agentcore_client,
            mode=CodeMode.CODE,
            system_prompt=(
                "You are a coding assistant. Always use the execute_code tool. "
                "Break every problem into many small steps, each requiring a separate code execution."
            ),
            max_tool_calls=1,
        )
        try:
            action = Action(
                message="Compute 1+1, then 2+2, then 3+3, then 4+4, then 5+5, each in a separate code execution."
            )
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
            assert result.observation.metrics["tool_calls"] >= 1
        finally:
            await env.cleanup()
