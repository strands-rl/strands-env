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

"""Unit tests for the Terminal-Bench (Harbor) evaluators."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("harbor", reason="harbor>=0.13.2 required to import the terminal_bench benchmark module")

from strands_env.eval import get_benchmark, list_benchmarks  # noqa: E402
from strands_env.eval.benchmarks.terminal_bench import TerminalBench21Evaluator  # noqa: E402


def test_terminal_bench_variants_registered():
    """All three Terminal-Bench variants are registered for discovery."""
    benchmarks = list_benchmarks()
    assert {"terminal-bench-1", "terminal-bench-2", "terminal-bench-2.1"} <= set(benchmarks)
    assert get_benchmark("terminal-bench-2.1") is TerminalBench21Evaluator


def test_terminal_bench_21_attributes():
    """Terminal-Bench-2.1 points at the upstream repo and nests tasks under `tasks/`."""
    assert TerminalBench21Evaluator.git_url == "https://github.com/harbor-framework/terminal-bench-2-1.git"
    assert TerminalBench21Evaluator.tasks_subdir == "tasks"


def test_data_dir_derived_from_benchmark_name(tmp_path: Path):
    """`data_dir` is derived from `benchmark_name` as `./data/<benchmark_name>`."""
    evaluator = TerminalBench21Evaluator(env_factory=lambda: None, output_path=tmp_path / "results.jsonl")
    assert evaluator.data_dir == Path("data") / "terminal-bench-2.1"


def test_terminal_bench_21_scans_tasks_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`load_dataset` scans `data_dir/tasks`, skipping the sibling `configs/` dir and dotfiles."""
    (tmp_path / "tasks" / "alpha").mkdir(parents=True)
    (tmp_path / "tasks" / "beta").mkdir()
    (tmp_path / "tasks" / ".hidden").mkdir()  # dotfiles are skipped
    (tmp_path / "configs").mkdir()  # leaderboard configs live outside tasks/ and must be ignored

    # data_dir is a read-only derived property; shadow it on the class to point at the fixture.
    monkeypatch.setattr(TerminalBench21Evaluator, "data_dir", tmp_path)  # exists, so no download
    evaluator = TerminalBench21Evaluator(env_factory=lambda: None, output_path=tmp_path / "out" / "results.jsonl")
    monkeypatch.setattr(evaluator, "_load_single_task", lambda task_dir: task_dir.name)

    assert evaluator.load_dataset() == ["alpha", "beta"]
