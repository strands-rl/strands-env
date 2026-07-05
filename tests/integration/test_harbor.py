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

"""Integration tests for HarborEnv with a real SGLang model.

Requires:
- A running SGLang server (default: http://localhost:30000)
- Docker daemon running
- harbor>=0.1.43 (`pip install harbor`)
"""

import shutil
import subprocess

import pytest

pytest.importorskip("harbor", reason="harbor>=0.1.43 required for harbor env integration tests")

from strands_env.core.types import TerminationReason
from strands_env.environments.harbor import HarborEnv, HarborTask

from .conftest import assert_rollout, assert_successful_rollout, assert_token_usage

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
    task = tmp_path_factory.mktemp("harbor_task")
    # harbor's Verifier reads the bundle spec; empty task.toml = all defaults, and
    # instruction.md is required by the bundle contract (unused here — tests inject messages).
    (task / "task.toml").write_text("")
    (task / "instruction.md").write_text("Run the command the user asks for.")

    env_dir = task / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(f"FROM ubuntu:22.04\nRUN mkdir -p {verifier_dir}\n")

    tests_dir = task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(f"#!/bin/bash\necho '1' > {verifier_dir}/reward.txt\n")

    return task


@pytest.fixture
def harbor_env(model_factory):
    """HarborEnv — capability only; each rollout() builds and tears down its own sandbox."""
    return HarborEnv(model_factory=model_factory)


@pytest.fixture
def make_task(task_dir, tmp_path):
    """Build a `HarborTask` for the fixture task bundle."""

    def _make(message: str) -> HarborTask:
        return HarborTask(
            message=message,
            task_id="test-task",
            task_dir=str(task_dir),
            trial_dir=str(tmp_path / "trial"),
        )

    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHarborEnv:
    async def test_rollout_with_docker_reward(self, harbor_env, make_task):
        """Full pipeline: agent runs command in Docker, result is complete, reward comes from test.sh."""
        result = await harbor_env.rollout(make_task("Run 'echo hello world' in the terminal."))

        assert_successful_rollout(result)
        assert_rollout(result)
        assert_token_usage(result)
        assert result.metrics["per_tool_metrics"]["execute_command"]["calls"] >= 1
        # The verifier ran and delivered a reward file to the host trial dir. status is the
        # pipeline assertion (deterministic); the reward VALUE is agent quality (stochastic) —
        # this line stays red if artifact delivery breaks (e.g. the harbor 0.13 mount regression).
        assert result.reward_result is not None
        assert result.reward_result.info["status"] == "success", result.reward_result.info

        # Reward: test.sh always writes 1 to reward.txt, validating the full pipeline
        # (upload tests → run test.sh → download results → parse reward)
        assert result.reward_result is not None
        assert result.reward_result.reward == 1.0

    async def test_multi_turn_conversation(self, harbor_env, make_task):
        """Agent uses conversation history from a prior turn to maintain context."""
        result1 = await harbor_env.rollout(make_task("Run 'echo hello' in the terminal."))
        assert result1.termination_reason == TerminationReason.TASK_COMPLETE

        task2 = make_task("Now run 'echo world'.")
        task2.conversation_history = result1.messages
        result2 = await harbor_env.rollout(task2)
        assert result2.termination_reason == TerminationReason.TASK_COMPLETE

    async def test_tool_iteration_limit(self, model_factory, task_dir, tmp_path):
        """max_tool_iters terminates the agent after the specified number of tool rounds."""
        env = HarborEnv(model_factory=model_factory, system_prompt=FORCE_TOOL_PROMPT, max_tool_iters=1)
        task = HarborTask(
            message=MANY_STEPS_PROMPT, task_id="test-iter-limit", task_dir=str(task_dir), trial_dir=str(tmp_path / "t1")
        )
        result = await env.rollout(task)

        assert result.termination_reason == TerminationReason.MAX_TOOL_ITERATIONS_REACHED
        assert result.metrics["tool_iters"] <= 1

    async def test_max_tool_calls_limit(self, model_factory, task_dir, tmp_path):
        """max_tool_calls terminates the agent after the specified total tool invocations."""
        env = HarborEnv(model_factory=model_factory, system_prompt=FORCE_TOOL_PROMPT, max_tool_calls=1)
        task = HarborTask(
            message=MANY_STEPS_PROMPT,
            task_id="test-calls-limit",
            task_dir=str(task_dir),
            trial_dir=str(tmp_path / "t2"),
        )
        result = await env.rollout(task)

        assert result.termination_reason == TerminationReason.MAX_TOOL_CALLS_REACHED
        assert result.metrics["tool_calls"] >= 1
