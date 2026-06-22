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

"""Unit tests for evaluation module (evaluator + metrics)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from strands_sglang import Rollout

from strands_env.core import RewardResult, RolloutResult, Task, TaskContext
from strands_env.eval import EvalSample, Evaluator
from strands_env.eval.metrics import compute_pass_at_k

from ..conftest import make_sample

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    async def test_factory_mode(self, mock_env, tmp_path):
        """Factory mode: reset/rollout/cleanup called for each sample."""
        mock_env.rollout.return_value = RolloutResult()

        async def factory(task):
            return mock_env

        tasks = [Task(id=f"p{i}", message=f"q{i}", context=TaskContext()) for i in range(3)]

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(tasks)

        assert mock_env.reset.await_count == 3
        assert mock_env.rollout.await_count == 3
        assert mock_env.cleanup.await_count == 3
        assert len(results) == 3
        assert sum(len(samples) for samples in results.values()) == 3

    async def test_n_samples_per_prompt_duplication(self, mock_env, tmp_path):
        """Each task is duplicated n_samples_per_prompt times."""
        mock_env.rollout.return_value = RolloutResult()

        async def factory(task):
            return mock_env

        tasks = [Task(id="p1", message="q", context=TaskContext())]

        evaluator = Evaluator(env_factory=factory, n_samples_per_prompt=5, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(tasks)

        assert mock_env.rollout.await_count == 5
        assert len(results) == 1
        assert "p1" in results
        assert len(results["p1"]) == 5

        sample_ids = [s.task.id for s in results["p1"]]
        assert len(set(sample_ids)) == 5

    async def test_factory_receives_task(self, tmp_path):
        """Factory receives the task for per-sample configuration."""
        received_tasks = []

        async def factory(task):
            received_tasks.append(task)
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(return_value=RolloutResult())
            env.cleanup = AsyncMock()
            return env

        tasks = [Task(message="q1"), Task(message="q2")]
        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        await evaluator.run(tasks)

        assert len(received_tasks) == 2
        assert received_tasks[0].message == "q1"
        assert received_tasks[1].message == "q2"

    async def test_max_concurrency(self, tmp_path):
        """max_concurrency limits concurrent env calls."""
        import asyncio

        concurrent_count = 0
        max_concurrent = 0

        async def mock_rollout(task):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return RolloutResult()

        async def factory(task):
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = mock_rollout
            env.cleanup = AsyncMock()
            return env

        tasks = [Task(message=f"q{i}") for i in range(10)]

        evaluator = Evaluator(env_factory=factory, max_concurrency=3, output_path=tmp_path / "results.jsonl")
        await evaluator.run(tasks)

        assert max_concurrent <= 3

    async def test_empty_tasks(self, mock_env, tmp_path):
        """Empty tasks produces empty results."""

        async def factory(task):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([])

        assert results == {}
        mock_env.rollout.assert_not_awaited()

    async def test_rollout_stripped_by_default(self, tmp_path):
        """Token rollout is stripped from results when keep_rollout=False (default)."""
        rollout = Rollout()
        rollout.add_prompt([1])
        rollout.add_response([2, 3], logprobs=[-0.5, -0.3])

        async def factory(task):
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(
                return_value=RolloutResult(rollout=rollout),
            )
            env.cleanup = AsyncMock()
            return env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        assert results["p1"][0].result.rollout is None

    async def test_cleanup_called_on_rollout_error(self, tmp_path):
        """env.cleanup() is called even when env.rollout() raises."""
        cleanup_called = False

        async def factory(task):
            nonlocal cleanup_called
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(side_effect=RuntimeError("rollout failed"))

            async def track_cleanup():
                nonlocal cleanup_called
                cleanup_called = True

            env.cleanup = track_cleanup
            return env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        sample = await evaluator.evaluate_sample(Task(id="err", message="q", context=TaskContext()))
        assert sample.aborted
        assert cleanup_called


# ---------------------------------------------------------------------------
# Checkpoint/Resume
# ---------------------------------------------------------------------------


class TestCheckpoint:
    async def test_saves_checkpoint(self, mock_env, tmp_path):
        """Results saved to output_path."""
        mock_env.rollout.return_value = RolloutResult()
        output_path = tmp_path / "results.jsonl"

        async def factory(task):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator.run([Task(id="s1", message="q1", context=TaskContext())])

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1

    async def test_resumes_from_checkpoint(self, mock_env, tmp_path):
        """Skips already-completed samples on resume."""
        mock_env.rollout.return_value = RolloutResult()
        output_path = tmp_path / "results.jsonl"

        async def factory(task):
            return mock_env

        # First run - complete s1
        evaluator1 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator1.run([Task(id="s1", message="q1", context=TaskContext())])
        assert mock_env.rollout.await_count == 1

        # Second run - s1 skipped, s2 processed
        mock_env.rollout.reset_mock()
        evaluator2 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results = await evaluator2.run(
            [
                Task(id="s1", message="q1", context=TaskContext()),
                Task(id="s2", message="q2", context=TaskContext()),
            ]
        )

        assert mock_env.rollout.await_count == 1  # Only s2 was processed
        assert len(results) == 2
        assert sum(len(samples) for samples in results.values()) == 2


# ---------------------------------------------------------------------------
# validate_sample / aborted
# ---------------------------------------------------------------------------


class TestValidateSample:
    async def test_aborted_samples_excluded_from_metrics(self, tmp_path):
        """Entire prompt is excluded from metrics if any sample is aborted."""
        results = {
            "p1": [make_sample(1.0, 0), make_sample(0.0, 1, aborted=True)],
            "p2": [make_sample(1.0, 2), make_sample(1.0, 3)],
            "p3": [make_sample(0.0, 4, aborted=True)],
        }

        async def factory(task):
            return MagicMock()

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        metrics = evaluator.compute_metrics(results, log=False)

        # Only p2 contributes → pass@1 = 1.0
        assert metrics["pass@1"] == 1.0

    async def test_aborted_samples_retried_on_resume(self, tmp_path):
        """Aborted samples are retried on resume (not added to completed_ids)."""
        rollout_count = 0

        class AbortingEvaluator(Evaluator):
            def validate_sample(self, sample):
                return False

        async def factory(task):
            nonlocal rollout_count
            rollout_count += 1
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(return_value=RolloutResult())
            env.cleanup = AsyncMock()
            return env

        output_path = tmp_path / "results.jsonl"

        # First run
        eval1 = AbortingEvaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results1 = await eval1.run([Task(id="s1", message="q1", context=TaskContext())])
        assert rollout_count == 1
        assert results1["s1"][0].aborted is True

        # Second run — s1 retried
        rollout_count = 0
        eval2 = AbortingEvaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results2 = await eval2.run([Task(id="s1", message="q1", context=TaskContext())])
        assert rollout_count == 1
        assert results2["s1"][0].aborted is True

    async def test_aborted_count_in_log(self, tmp_path, caplog):
        """Skipped prompts and aborted count appear in metric log output."""
        results = {
            "p1": [make_sample(1.0, 0), make_sample(0.0, 1, aborted=True)],
            "p2": [make_sample(1.0, 2)],
        }

        async def factory(task):
            return MagicMock()

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")

        with caplog.at_level(logging.INFO):
            evaluator.compute_metrics(results, log=True)

        assert "Skipped 1 prompts due to aborted samples" in caplog.text

    async def test_custom_validate_sample(self, tmp_path):
        """Subclass can override validate_sample to abort specific samples."""

        class RewardCheckEvaluator(Evaluator):
            def validate_sample(self, sample):
                return sample.result.reward_result is not None

        call_count = 0

        async def factory(task):
            nonlocal call_count
            call_count += 1
            reward = RewardResult(reward=1.0) if call_count == 2 else None
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(return_value=RolloutResult(reward_result=reward))
            env.cleanup = AsyncMock()
            return env

        evaluator = RewardCheckEvaluator(
            env_factory=factory,
            n_samples_per_prompt=2,
            max_concurrency=1,
            output_path=tmp_path / "results.jsonl",
        )
        results = await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        assert len(results["p1"]) == 2
        aborted = [s for s in results["p1"] if s.aborted]
        valid = [s for s in results["p1"] if not s.aborted]
        assert len(aborted) == 1
        assert len(valid) == 1
        assert valid[0].result.reward_result.reward == 1.0


# ---------------------------------------------------------------------------
# Default compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    async def test_default_metrics(self, tmp_path):
        """Default metric_fns includes pass@k."""

        async def factory(task):
            env = MagicMock()
            env.reset = AsyncMock()
            env.rollout = AsyncMock(
                return_value=RolloutResult(
                    reward_result=RewardResult(reward=1.0),
                )
            )
            env.cleanup = AsyncMock()
            return env

        evaluator = Evaluator(env_factory=factory, n_samples_per_prompt=3, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        metrics = evaluator.compute_metrics(results)
        assert "pass@1" in metrics
        assert "pass@2" in metrics
        assert "pass@3" in metrics
        assert metrics["pass@1"] == 1.0


# ---------------------------------------------------------------------------
# pass@k metric
# ---------------------------------------------------------------------------


class TestPassAtK:
    def test_empty_samples(self):
        result = compute_pass_at_k({}, k_values=[1])
        assert result == {"pass@1": 0.0}

    def test_all_correct(self):
        results = {"p1": [make_sample(1.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == 1.0

    def test_none_correct(self):
        results = {"p1": [make_sample(0.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == 0.0

    def test_multiple_problems(self):
        results = {
            "p1": [make_sample(1.0, 0), make_sample(1.0, 1)],
            "p2": [make_sample(0.0, 0), make_sample(0.0, 1)],
        }
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == pytest.approx(0.5)

    def test_multiple_k_values(self):
        results = {"p1": [make_sample(1.0 if i == 0 else 0.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1, 5])
        assert result["pass@1"] == pytest.approx(0.2)
        assert result["pass@5"] == pytest.approx(1.0)

    def test_custom_reward_threshold(self):
        results = {"p1": [make_sample(0.5, 0)]}
        result = compute_pass_at_k(results, k_values=[1], reward_threshold=1.0)
        assert result["pass@1"] == 0.0
        result = compute_pass_at_k(results, k_values=[1], reward_threshold=0.5)
        assert result["pass@1"] == 1.0

    def test_k_larger_than_n_skipped(self):
        results = {"p1": [make_sample(1.0, 0), make_sample(1.0, 1)]}
        result = compute_pass_at_k(results, k_values=[5])
        assert result["pass@5"] == 0.0

    def test_none_reward_handled(self):
        """Samples with None reward are treated as incorrect."""
        sample = EvalSample(
            task=Task(id="p1_0", message="q", context=TaskContext()),
            result=RolloutResult(reward_result=None),
        )
        result = compute_pass_at_k({"p1": [sample]}, k_values=[1])
        assert result["pass@1"] == 0.0

    def test_half_correct(self):
        results = {"p1": [make_sample(1.0 if i < 5 else 0.0, i) for i in range(10)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == pytest.approx(0.5)

    def test_one_correct_out_of_ten(self):
        results = {"p1": [make_sample(1.0 if i == 0 else 0.0, i) for i in range(10)]}
        result = compute_pass_at_k(results, k_values=[1, 5])
        assert result["pass@1"] == pytest.approx(0.1)
        assert result["pass@5"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Distributed evaluation (env_actor_pool)
# ---------------------------------------------------------------------------


class TestDistributedEvaluation:
    async def test_pool_rollout_called(self, tmp_path):
        """When env_actor_pool is provided, pool.rollout is used instead of env_factory."""
        mock_pool = MagicMock()
        mock_pool.rollout = AsyncMock(return_value=RolloutResult())

        evaluator = Evaluator(env_actor_pool=mock_pool, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        mock_pool.rollout.assert_awaited_once()
        assert len(results) == 1

    async def test_env_factory_not_called_when_pool_set(self, tmp_path):
        """env_factory is not called when env_actor_pool is provided."""
        mock_pool = MagicMock()
        mock_pool.rollout = AsyncMock(return_value=RolloutResult())
        factory_called = False

        async def factory(task):
            nonlocal factory_called
            factory_called = True

        evaluator = Evaluator(env_factory=factory, env_actor_pool=mock_pool, output_path=tmp_path / "results.jsonl")
        await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        assert not factory_called

    async def test_pool_error_aborts_sample(self, tmp_path):
        """Exceptions from pool.rollout produce aborted samples."""
        mock_pool = MagicMock()
        mock_pool.rollout = AsyncMock(side_effect=RuntimeError("remote error"))

        evaluator = Evaluator(env_actor_pool=mock_pool, output_path=tmp_path / "results.jsonl")
        sample = await evaluator.evaluate_sample(Task(id="err", message="q", context=TaskContext()))

        assert sample.aborted

    async def test_pool_rollout_stripped(self, tmp_path):
        """Token rollout is stripped from pool results when keep_rollout=False."""
        rollout = Rollout()
        rollout.add_prompt([1])
        rollout.add_response([2, 3], logprobs=[-0.5, -0.3])
        mock_pool = MagicMock()
        mock_pool.rollout = AsyncMock(return_value=RolloutResult(rollout=rollout))

        evaluator = Evaluator(env_actor_pool=mock_pool, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([Task(id="p1", message="q", context=TaskContext())])

        assert results["p1"][0].result.rollout is None

    async def test_init_requires_factory_or_pool(self, tmp_path):
        """ValueError raised when neither env_factory nor env_actor_pool is provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            Evaluator(output_path=tmp_path / "results.jsonl")
