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

"""Unit tests for `HarborConfig` type resolution.

Regression guard: `HarborConfig` (with its nested `PrebakedE2BConfig` and the
inherited `trace_attributes`) is embedded as a Pydantic field by the
terminal-bench / swebench task contexts. Pydantic resolves those hints at
runtime, so a TYPE_CHECKING-only type in any `HarborConfig` field breaks dataset
loading with a `NameError`.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import BaseModel

pytest.importorskip("harbor", reason="harbor>=0.13.2 required for HarborConfig")

from strands_env.environments.harbor.env import HarborConfig  # noqa: E402
from strands_env.environments.harbor.task import HarborTask  # noqa: E402


def test_harbor_config_type_hints_resolve():
    """`HarborConfig` annotations resolve at runtime (incl. `prebaked_e2b_config`)."""
    hints = typing.get_type_hints(HarborConfig)
    assert "prebaked_e2b_config" in hints
    assert "backend" in hints
    assert "trace_attributes" in hints  # inherited from EnvironmentConfig


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
