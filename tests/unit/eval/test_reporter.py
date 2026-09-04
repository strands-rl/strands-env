from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from strands_env.core import RewardResult, RolloutResult, Task
from strands_env.eval import CompositeReporter, EvalReporter, EvalSample, Evaluator, LocalReporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples() -> list[tuple[str, EvalSample]]:
    """Build a small list of (prompt_id, sample) pairs for testing."""
    return [
        (
            "p1",
            EvalSample(
                task=Task(id="p1_0", message="q1"), result=RolloutResult(reward_result=RewardResult(reward=1.0))
            ),
        ),
        (
            "p2",
            EvalSample(
                task=Task(id="p2_0", message="q2"), result=RolloutResult(reward_result=RewardResult(reward=0.5))
            ),
        ),
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().strip().split("\n")]


# ---------------------------------------------------------------------------
# LocalReporter
# ---------------------------------------------------------------------------


class TestLocalReporter:
    def test_log_sample_then_flush_writes_jsonl(self, tmp_path: Path):
        """log_sample streams each sample; flush syncs. Output matches the old save_results format."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)

        for prompt_id, sample in _make_samples():
            reporter.log_sample(prompt_id, sample)
        reporter.flush()

        assert output_path.exists()
        rows = _read_jsonl(output_path)
        assert len(rows) == 2
        assert rows[0]["prompt_id"] == "p1"
        assert rows[0]["task"]["id"] == "p1_0"
        assert rows[0]["result"]["reward_result"]["reward"] == 1.0

    def test_log_sample_survives_an_unpaired_surrogate(self, tmp_path: Path):
        """A lone surrogate in model output must not fail the write."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)
        sample = EvalSample(
            task=Task(id="p1_0", message="half a math-italic code point: \ud835"),
            result=RolloutResult(reward_result=RewardResult(reward=1.0)),
        )

        reporter.log_sample("p1", sample)
        reporter.flush()

        rows = _read_jsonl(output_path)
        assert rows[0]["task"]["message"] == "half a math-italic code point: \ud835"

    def test_writes_bytes_as_base64(self, tmp_path: Path):
        """Bedrock returns redacted reasoning as bytes, which json.dumps rejects."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)
        redacted = {"role": "assistant", "content": [{"reasoningContent": {"redactedContent": b"\x00\x01"}}]}
        sample = EvalSample(task=Task(id="p1_0", message="q1"), result=RolloutResult(messages=[redacted]))

        reporter.log_sample("p1", sample)
        reporter.flush()
        rows = _read_jsonl(output_path)
        assert rows[0]["result"]["messages"][0]["content"][0]["reasoningContent"]["redactedContent"] == "AAE="

        reporter.rewrite({"p1": [sample]})
        rows = _read_jsonl(output_path)
        assert rows[0]["result"]["messages"][0]["content"][0]["reasoningContent"]["redactedContent"] == "AAE="

    def test_log_sample_creates_parent_dirs(self, tmp_path: Path):
        """The first log_sample creates parent directories if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "results.jsonl"
        reporter = LocalReporter(output_path)

        prompt_id, sample = _make_samples()[0]
        reporter.log_sample(prompt_id, sample)
        reporter.flush()

        assert output_path.exists()

    def test_log_metrics_writes_json(self, tmp_path: Path):
        """log_metrics() writes metrics.json next to the output path."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)

        reporter.log_metrics({"pass@1": 0.75, "pass@2": 0.90})

        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        assert json.loads(metrics_path.read_text()) == {"pass@1": 0.75, "pass@2": 0.90}

    def test_log_metadata_writes_json(self, tmp_path: Path):
        """log_metadata() writes metadata.json next to the output path."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)

        reporter.log_metadata({"benchmark": "aime-2024", "backend": "sglang", "n_samples": 30})

        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()
        data = json.loads(metadata_path.read_text())
        assert data["benchmark"] == "aime-2024"
        assert data["backend"] == "sglang"

    def test_flush_before_any_sample_is_safe(self, tmp_path: Path):
        """flush() with no logged samples is a no-op (no file handle open yet)."""
        reporter = LocalReporter(tmp_path / "results.jsonl")
        reporter.flush()  # must not raise

    async def test_publish_closes_handle(self, tmp_path: Path):
        """publish() flushes and closes the open file handle."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)
        prompt_id, sample = _make_samples()[0]
        reporter.log_sample(prompt_id, sample)

        await reporter.publish()

        assert _read_jsonl(output_path)[0]["prompt_id"] == "p1"

    def test_rewrite_purges_stale_entries(self, tmp_path: Path):
        """rewrite() reconciles the file to exactly the given results, dropping stale rows."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)

        # A stale aborted row is on disk (as if left by a prior run), then a good row streams in.
        aborted = EvalSample(task=Task(id="p1_0", message="q1"), result=RolloutResult(), aborted=True)
        reporter.log_sample("p1", aborted)
        reporter.flush()

        # On resume the evaluator keeps only the valid sample and reconciles the checkpoint.
        good = EvalSample(
            task=Task(id="p1_0", message="q1"), result=RolloutResult(reward_result=RewardResult(reward=1.0))
        )
        reporter.rewrite({"p1": [good]})

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["task"]["id"] == "p1_0"
        assert rows[0]["aborted"] is False

    def test_rewrite_then_log_sample_appends(self, tmp_path: Path):
        """After rewrite reopens the file, subsequent log_sample calls append (not overwrite)."""
        output_path = tmp_path / "results.jsonl"
        reporter = LocalReporter(output_path)
        kept, _ = _make_samples()[0]  # ("p1", sample)
        reporter.rewrite({kept: [_make_samples()[0][1]]})

        prompt_id, sample = _make_samples()[1]  # ("p2", sample)
        reporter.log_sample(prompt_id, sample)
        reporter.flush()

        rows = _read_jsonl(output_path)
        assert [r["prompt_id"] for r in rows] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# CompositeReporter
# ---------------------------------------------------------------------------


class TestCompositeReporter:
    def test_fan_out_log_sample(self):
        """log_sample fans out to all reporters."""
        r1 = MagicMock(spec=EvalReporter)
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])
        sample = EvalSample(task=Task(id="s1", message="q"), result=RolloutResult())

        composite.log_sample("p1", sample)

        r1.log_sample.assert_called_once_with("p1", sample)
        r2.log_sample.assert_called_once_with("p1", sample)

    def test_fan_out_flush(self):
        """flush fans out to all reporters (argless)."""
        r1 = MagicMock(spec=EvalReporter)
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])

        composite.flush()

        r1.flush.assert_called_once_with()
        r2.flush.assert_called_once_with()

    def test_fan_out_rewrite(self):
        """rewrite fans out to all reporters."""
        r1 = MagicMock(spec=EvalReporter)
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])
        results = {"p1": []}

        composite.rewrite(results)

        r1.rewrite.assert_called_once_with(results)
        r2.rewrite.assert_called_once_with(results)

    def test_fan_out_log_metrics(self):
        """log_metrics fans out to all reporters."""
        r1 = MagicMock(spec=EvalReporter)
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])
        metrics = {"pass@1": 0.8}

        composite.log_metrics(metrics)

        r1.log_metrics.assert_called_once_with(metrics)
        r2.log_metrics.assert_called_once_with(metrics)

    def test_fan_out_log_metadata(self):
        """log_metadata fans out to all reporters."""
        r1 = MagicMock(spec=EvalReporter)
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])
        metadata = {"benchmark": "aime-2024"}

        composite.log_metadata(metadata)

        r1.log_metadata.assert_called_once_with(metadata)
        r2.log_metadata.assert_called_once_with(metadata)

    async def test_fan_out_publish(self):
        """publish fans out to all reporters."""
        r1 = MagicMock(spec=EvalReporter)
        r1.publish = AsyncMock()
        r2 = MagicMock(spec=EvalReporter)
        r2.publish = AsyncMock()
        composite = CompositeReporter([r1, r2])

        await composite.publish()

        r1.publish.assert_awaited_once()
        r2.publish.assert_awaited_once()

    def test_error_isolation_flush(self):
        """If one reporter raises in flush, others still execute."""
        r1 = MagicMock(spec=EvalReporter)
        r1.flush.side_effect = RuntimeError("r1 broken")
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])

        composite.flush()

        r2.flush.assert_called_once_with()

    def test_error_isolation_log_sample(self):
        """If one reporter raises in log_sample, others still execute."""
        r1 = MagicMock(spec=EvalReporter)
        r1.log_sample.side_effect = RuntimeError("r1 broken")
        r2 = MagicMock(spec=EvalReporter)
        composite = CompositeReporter([r1, r2])
        sample = EvalSample(task=Task(id="s1", message="q"), result=RolloutResult())

        composite.log_sample("p1", sample)

        r2.log_sample.assert_called_once_with("p1", sample)

    async def test_error_isolation_publish(self):
        """If one reporter raises in publish, others still execute."""
        r1 = MagicMock(spec=EvalReporter)
        r1.publish = AsyncMock(side_effect=RuntimeError("r1 broken"))
        r2 = MagicMock(spec=EvalReporter)
        r2.publish = AsyncMock()
        composite = CompositeReporter([r1, r2])

        await composite.publish()

        r2.publish.assert_awaited_once()


# ---------------------------------------------------------------------------
# Evaluator + Reporter integration
# ---------------------------------------------------------------------------


class TestEvaluatorReporterIntegration:
    async def test_default_reporter_is_local(self, tmp_path: Path):
        """Evaluator without explicit reporter creates a LocalReporter."""

        async def factory():
            env = MagicMock()
            env.rollout = AsyncMock(return_value=RolloutResult())
            return env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        assert isinstance(evaluator.reporter, LocalReporter)

    async def test_custom_reporter_injected(self, tmp_path: Path):
        """An explicitly provided reporter receives log_sample and flush calls."""
        mock_reporter = MagicMock(spec=EvalReporter)

        async def factory():
            env = MagicMock()
            env.rollout = AsyncMock(return_value=RolloutResult())
            return env

        evaluator = Evaluator(
            env_factory=factory,
            output_path=tmp_path / "results.jsonl",
            reporter=mock_reporter,
            save_interval=1,
        )
        await evaluator.run([Task(id="p1", message="q1"), Task(id="p2", message="q2")])

        assert mock_reporter.log_sample.call_count == 2
        assert mock_reporter.flush.call_count >= 2

    async def test_backward_compat_output(self, tmp_path: Path):
        """Default LocalReporter produces the same JSONL content as the old hardcoded save_results."""
        output_path = tmp_path / "results.jsonl"

        async def factory():
            env = MagicMock()
            env.rollout = AsyncMock(return_value=RolloutResult(reward_result=RewardResult(reward=1.0)))
            return env

        evaluator = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator.run([Task(id="p1", message="q1")])

        assert output_path.exists()
        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["prompt_id"] == "p1"
        assert rows[0]["task"]["id"] == "p1_0"
        assert rows[0]["result"]["reward_result"]["reward"] == 1.0

    async def test_resume_after_abort_has_no_duplicate(self, tmp_path: Path):
        """A sample that aborts then succeeds on resume leaves exactly one (non-aborted) row."""
        output_path = tmp_path / "results.jsonl"

        async def failing_factory():
            env = MagicMock()
            env.rollout = AsyncMock(side_effect=RuntimeError("boom"))  # -> aborted sample
            return env

        # First run: the single sample aborts and is written to the checkpoint.
        evaluator = Evaluator(env_factory=failing_factory, output_path=output_path, save_interval=1)
        await evaluator.run([Task(id="p1", message="q1")])
        assert any(r["aborted"] for r in _read_jsonl(output_path))

        async def good_factory():
            env = MagicMock()
            env.rollout = AsyncMock(return_value=RolloutResult(reward_result=RewardResult(reward=1.0)))
            return env

        # Resume: the aborted sample is retried and now succeeds. The stale aborted row must be gone.
        evaluator = Evaluator(env_factory=good_factory, output_path=output_path, save_interval=1)
        await evaluator.run([Task(id="p1", message="q1")])

        rows = _read_jsonl(output_path)
        assert len(rows) == 1  # no duplicate task entry
        assert rows[0]["task"]["id"] == "p1_0"
        assert rows[0]["aborted"] is False
