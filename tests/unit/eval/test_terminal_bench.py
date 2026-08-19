from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("harbor", reason="harbor>=0.13.2 required to import the terminal_bench benchmark module")

from strands_env.eval import get_benchmark, list_benchmarks
from strands_env.eval.benchmarks.terminal_bench import TerminalBench21Evaluator


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


def _write_task_bundle(task_dir: Path, *, memory_mb: int | None) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do the thing\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("exit 0\n")
    memory_line = f"memory_mb = {memory_mb}\n" if memory_mb is not None else ""
    (task_dir / "task.toml").write_text(
        'schema_version = "1.1"\n'
        "artifacts = []\n"
        "\n"
        "[task]\n"
        f'name = "fixture/{task_dir.name}"\n'
        'description = "fixture"\n'
        "\n"
        "[verifier]\n"
        "timeout_sec = 60.0\n"
        "\n"
        "[environment]\n"
        'docker_image = "alpine:3"\n'
        f"{memory_line}"
    )


@pytest.mark.parametrize(
    ("declared", "expected"),
    [(2048, 4096), (None, None)],
    ids=["doubles_when_declared", "left_alone_when_absent"],
)
def test_memory_mb_doubling_tolerates_an_absent_value(
    tmp_path: Path, declared: int | None, expected: int | None
) -> None:
    """`memory_mb` is Optional upstream, so doubling it has to survive a task that omits it."""
    task_dir = tmp_path / "alpha"
    _write_task_bundle(task_dir, memory_mb=declared)
    evaluator = TerminalBench21Evaluator(env_factory=lambda: None, output_path=tmp_path / "out" / "results.jsonl")

    task = evaluator._load_single_task(task_dir)

    assert task.task_env_config.memory_mb == expected
