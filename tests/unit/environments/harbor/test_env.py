from __future__ import annotations

import typing
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

pytest.importorskip("harbor", reason="harbor>=0.13.2 required for HarborConfig")

from strands_env.environments.harbor import env as harbor_env
from strands_env.environments.harbor.env import HarborConfig, HarborEnv
from strands_env.environments.harbor.task import HarborTask


def test_harbor_config_type_hints_resolve():
    """`HarborConfig` annotations resolve at runtime (incl. `prebaked_e2b_config`)."""
    hints = typing.get_type_hints(HarborConfig)
    assert "prebaked_e2b_config" in hints
    assert "backend" in hints
    assert "start_jitter" in hints
    assert "system_prompt" in hints  # inherited from EnvironmentConfig


def test_harbor_config_embeds_in_pydantic_model():
    """`HarborConfig` can back a Pydantic field — its annotations must resolve at runtime (#141)."""

    class _Ctx(BaseModel):
        config: HarborConfig

    _Ctx.model_rebuild()
    ctx = _Ctx(config={"backend": "docker", "exec_timeout": 600})
    assert ctx.config["backend"] == "docker"


def test_harbor_task_round_trips():
    """`HarborTask` carries the per-sample payload and survives JSON transport."""
    task = HarborTask(message="fix the bug", task_id="t", task_dir="/d", trial_dir="/o")
    again = HarborTask.model_validate_json(task.model_dump_json())
    assert again.task_id == "t"
    assert again.trial_dir == "/o"


async def _reset_with(jitter_config: dict, tmp_path) -> MagicMock:
    """Run `reset()` on a docker-backed env and return the patched `asyncio.sleep` mock.

    The docker backend is used because its sandbox comes from a factory we can patch
    wholesale, so `reset()` runs end to end without Docker installed.
    """
    env = HarborEnv(model_factory=MagicMock(), backend="docker", **jitter_config)
    task = HarborTask(message="m", task_id="t", task_dir=str(tmp_path), trial_dir=str(tmp_path / "trial"))
    sandbox = MagicMock()
    sandbox.start = AsyncMock()

    with (
        patch.object(harbor_env.EnvironmentFactory, "create_environment", return_value=sandbox),
        patch.object(harbor_env.asyncio, "sleep", new=AsyncMock()) as sleep,
    ):
        await env.reset(task)

    sandbox.start.assert_awaited_once()
    return sleep


class TestStartJitter:
    """`start_jitter` staggers sandbox creation across concurrent episodes."""

    async def test_defaults_to_no_sleep(self, tmp_path):
        """Unset `start_jitter` must not add latency to a serial or low-concurrency run."""
        env = HarborEnv(model_factory=MagicMock(), backend="docker")
        assert env.start_jitter == 0.0
        sleep = await _reset_with({}, tmp_path)
        sleep.assert_not_awaited()

    async def test_sleeps_within_the_window(self, tmp_path):
        """A positive `start_jitter` sleeps somewhere in [0, jitter) before starting."""
        sleep = await _reset_with({"start_jitter": 30.0}, tmp_path)
        sleep.assert_awaited_once()
        assert 0 <= sleep.await_args.args[0] < 30.0

    async def test_zero_is_treated_as_disabled(self, tmp_path):
        """An explicit 0 disables the sleep rather than awaiting `sleep(0)`."""
        sleep = await _reset_with({"start_jitter": 0}, tmp_path)
        sleep.assert_not_awaited()

    def test_int_config_is_coerced_to_float(self):
        """`--env-config` JSON yields ints; they must not break the comparison or sleep."""
        env = HarborEnv(model_factory=MagicMock(), backend="docker", start_jitter=15)  # type: ignore[typeddict-item]
        assert env.start_jitter == 15.0
        assert isinstance(env.start_jitter, float)
