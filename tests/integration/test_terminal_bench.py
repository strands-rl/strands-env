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
from strands_env.environments.terminal_bench import TerminalBenchEnv

from .conftest import assert_successful_step, assert_token_observation, assert_token_usage

FORCE_TOOL_PROMPT = (
    "You are a terminal assistant. Always use execute_command. "
    "Break every task into many small steps, each in a separate command."
)

MANY_STEPS_PROMPT = "Run 'echo 1', then 'echo 2', then 'echo 3', then 'echo 4', then 'echo 5' one at a time."

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
    env = TerminalBenchEnv(
        model_factory=model_factory,
        task_id="test-task",
        task_dir=str(task_dir),
        trial_dir=str(tmp_path / "trial"),
    )
    await env.reset()
    yield env
    await env.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTerminalBench:
    async def test_step_with_docker_reward(self, terminal_bench_env):
        """Full pipeline: agent runs command in Docker, observation is complete, reward comes from test.sh."""
        result = await terminal_bench_env.step(Action(message="Run 'echo hello world' in the terminal."))

        assert_successful_step(result)
        assert_token_observation(result)
        assert_token_usage(result)
        assert result.observation.metrics["per_tool_metrics"]["execute_command"]["calls"] >= 1

        # Reward: test.sh always writes 1 to reward.txt, validating the full pipeline
        # (upload tests → run test.sh → download results → parse reward)
        assert result.reward is not None
        assert result.reward.reward == 1.0

    async def test_multi_turn_conversation(self, terminal_bench_env):
        """Agent uses conversation history from a prior turn to maintain context."""
        result1 = await terminal_bench_env.step(Action(message="Run 'echo hello' in the terminal."))
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        result2 = await terminal_bench_env.step(
            Action(
                message="Now run 'echo world'.",
                task_context=TaskContext(conversation_history=result1.observation.messages),
            ),
        )
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_tool_iteration_limit(self, model_factory, task_dir, tmp_path):
        """max_tool_iters terminates the agent after the specified number of tool rounds."""
        env = TerminalBenchEnv(
            model_factory=model_factory,
            task_id="test-iter-limit",
            task_dir=str(task_dir),
            trial_dir=str(tmp_path / "trial"),
            system_prompt=FORCE_TOOL_PROMPT,
            max_tool_iters=1,
        )
        try:
            await env.reset()
            result = await env.step(Action(message=MANY_STEPS_PROMPT))

            assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
            assert result.observation.metrics["tool_iters"] <= 1
        finally:
            await env.cleanup()

    async def test_max_tool_calls_limit(self, model_factory, task_dir, tmp_path):
        """max_tool_calls terminates the agent after the specified total tool invocations."""
        env = TerminalBenchEnv(
            model_factory=model_factory,
            task_id="test-calls-limit",
            task_dir=str(task_dir),
            trial_dir=str(tmp_path / "trial"),
            system_prompt=FORCE_TOOL_PROMPT,
            max_tool_calls=1,
        )
        try:
            await env.reset()
            result = await env.step(Action(message=MANY_STEPS_PROMPT))

            assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
            assert result.observation.metrics["tool_calls"] >= 1
        finally:
            await env.cleanup()
