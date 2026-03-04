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

"""Integration tests for TerminalBenchEnv with a real SGLang model.

Requires:
- A running SGLang server (default: http://localhost:30000)
- Docker daemon running
- harbor>=0.1.43 (`pip install harbor`)
"""

import shutil
import subprocess

import pytest

pytest.importorskip("harbor", reason="harbor>=0.1.43 required for terminal_bench integration tests")

from strands_env.core.types import Action, TaskContext, TerminationReason
from strands_env.environments.terminal_bench import TerminalBenchConfig, TerminalBenchEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def docker_available():
    """Skip all tests if Docker daemon is not running."""
    if not shutil.which("docker"):
        pytest.skip("docker CLI not found")
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)  # noqa: S603, S607
        if result.returncode != 0:
            pytest.skip("Docker daemon not running")
    except subprocess.TimeoutExpired:
        pytest.skip("Docker daemon not responding")


@pytest.fixture(scope="session")
def task_dir(tmp_path_factory, docker_available):
    """Minimal task directory with a simple Dockerfile and always-passing test."""
    from harbor.models.trial.paths import EnvironmentPaths

    verifier_dir = EnvironmentPaths.verifier_dir
    task = tmp_path_factory.mktemp("terminal_bench_task")

    env_dir = task / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(f"FROM ubuntu:22.04\nRUN mkdir -p {verifier_dir}\n")

    tests_dir = task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(f"#!/bin/bash\necho '1' > {verifier_dir}/reward.txt\n")

    return task


@pytest.fixture
async def terminal_bench_env(model_factory, task_dir, tmp_path):
    """TerminalBenchEnv with Docker reset and cleanup."""
    config = TerminalBenchConfig(
        task_id="test-task",
        task_dir=task_dir,
        trial_dir=tmp_path / "trial",
    )
    env = TerminalBenchEnv(model_factory=model_factory, config=config)
    await env.reset()
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTerminalBench:
    async def test_step_completes(self, terminal_bench_env):
        """Agent can execute shell commands in Docker and complete normally."""
        action = Action(message="Run 'echo hello world' in the terminal.")
        result = await terminal_bench_env.step(action)

        assert result.termination_reason == TerminationReason.TASK_COMPLETE
        assert result.observation.messages
        assert result.observation.metrics["message_count"] > 0

    async def test_step_produces_token_observation(self, terminal_bench_env):
        """SGLang model produces token-level observations for TITO training."""
        action = Action(message="List the contents of the root directory.")
        result = await terminal_bench_env.step(action)

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

    async def test_step_metrics(self, terminal_bench_env):
        """Step produces expected metric keys with correct structure."""
        action = Action(message="Check the current working directory with pwd.")
        result = await terminal_bench_env.step(action)

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
        # Per-tool metrics: execute_command should appear with correct structure
        per_tool = metrics.get("per_tool_metrics")
        assert per_tool is not None
        assert "execute_command" in per_tool
        tool_m = per_tool["execute_command"]
        assert tool_m["calls"] >= 1
        assert tool_m["successes"] >= 1
        assert "latency_s" in tool_m

    async def test_final_response(self, terminal_bench_env):
        """Observation provides final assistant response text."""
        action = Action(message="Run 'echo 42' and tell me the output.")
        result = await terminal_bench_env.step(action)

        response = result.observation.final_response
        assert response is not None
        assert len(response) > 0

    async def test_conversation_history(self, terminal_bench_env):
        """Multi-turn interaction with conversation history."""
        action1 = Action(message="Run 'echo hello' in the terminal.")
        result1 = await terminal_bench_env.step(action1)
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        all_messages = result1.observation.messages
        action2 = Action(
            message="Now run 'echo world'.",
            task_context=TaskContext(conversation_history=all_messages),
        )
        result2 = await terminal_bench_env.step(action2)
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_reward_computation(self, terminal_bench_env):
        """Default TerminalBenchRewardFunction computes reward via test.sh in Docker."""
        action = Action(message="Run 'echo hello' in the terminal.")
        result = await terminal_bench_env.step(action)

        # test.sh always writes 1 to reward.txt, validating the full pipeline:
        # upload tests → run test.sh → download results → parse reward
        assert result.reward is not None
        assert isinstance(result.reward.reward, float)
        assert result.reward.reward == 1.0


# ---------------------------------------------------------------------------
# Tests — tool limits
# ---------------------------------------------------------------------------


class TestToolLimit:
    async def test_tool_iteration_limit(self, model_factory, task_dir, tmp_path):
        """Environment respects max_tool_iters."""
        config = TerminalBenchConfig(
            task_id="test-iter-limit",
            task_dir=task_dir,
            trial_dir=tmp_path / "trial",
        )
        env = TerminalBenchEnv(
            model_factory=model_factory,
            config=config,
            system_prompt=(
                "You are a terminal assistant. Always use execute_command. "
                "Break every task into many small steps, each in a separate command."
            ),
            max_tool_iters=1,
        )
        try:
            await env.reset()
            action = Action(
                message="Run 'echo 1', then 'echo 2', then 'echo 3', then 'echo 4', then 'echo 5' one at a time."
            )
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            assert result.observation.metrics["tool_iters"] <= 1
        finally:
            await env.cleanup()

    async def test_max_tool_calls(self, model_factory, task_dir, tmp_path):
        """Environment respects max_tool_calls (distinct from max_tool_iters).

        Note: parallel tool calls within a single iteration may exceed the limit
        before the limiter fires, so we only assert on termination reason.
        """
        config = TerminalBenchConfig(
            task_id="test-calls-limit",
            task_dir=task_dir,
            trial_dir=tmp_path / "trial",
        )
        env = TerminalBenchEnv(
            model_factory=model_factory,
            config=config,
            system_prompt=(
                "You are a terminal assistant. Always use execute_command. "
                "Break every task into many small steps, each in a separate command."
            ),
            max_tool_calls=1,
        )
        try:
            await env.reset()
            action = Action(
                message="Run 'echo 1', then 'echo 2', then 'echo 3', then 'echo 4', then 'echo 5' one at a time."
            )
            result = await env.step(action)

            assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
            assert result.observation.metrics["tool_calls"] >= 1
        finally:
            await env.cleanup()
