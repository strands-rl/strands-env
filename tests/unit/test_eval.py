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

"""Unit tests for evaluation module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strands_env.core import Action, Environment, Observation, RewardResult, StepResult, TaskContext
from strands_env.eval import EvalSample, Evaluator
from strands_env.eval.benchmarks.aime import AIME2024Evaluator
from strands_env.eval.metrics import compute_pass_at_k

# ---------------------------------------------------------------------------
# EvalSample
# ---------------------------------------------------------------------------


class TestEvalSample:
    def test_basic_fields(self):
        step_result = StepResult(observation=Observation())
        sample = EvalSample(action=Action(message="test"), step_result=step_result)
        assert sample.action.message == "test"
        assert sample.step_result == step_result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock(spec=Environment)
        env.reset = AsyncMock()
        env.step = AsyncMock()
        env.cleanup = AsyncMock()
        return env

    async def test_factory_mode(self, mock_env, tmp_path):
        """Factory mode: reset/step/cleanup called for each sample."""
        mock_env.step.return_value = StepResult(observation=Observation())

        async def factory(action):
            return mock_env

        actions = [Action(message=f"q{i}", task_context=TaskContext(id=f"p{i}")) for i in range(3)]

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(actions)

        # With n_samples_per_prompt=1 (default), each action is run once
        assert mock_env.reset.await_count == 3
        assert mock_env.step.await_count == 3
        assert mock_env.cleanup.await_count == 3
        assert len(results) == 3  # 3 prompt_ids
        assert sum(len(samples) for samples in results.values()) == 3

    async def test_n_samples_per_prompt_duplication(self, mock_env, tmp_path):
        """Each action is duplicated n_samples_per_prompt times."""
        mock_env.step.return_value = StepResult(observation=Observation())

        async def factory(action):
            return mock_env

        actions = [Action(message="q", task_context=TaskContext(id="p1"))]

        evaluator = Evaluator(env_factory=factory, n_samples_per_prompt=5, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(actions)

        # 5 rollouts per problem
        assert mock_env.step.await_count == 5
        assert len(results) == 1  # One prompt_id key
        assert "p1" in results
        assert len(results["p1"]) == 5  # 5 samples for that problem

        # Each should have unique sample_id (stored in action.task_context.id)
        sample_ids = [s.action.task_context.id for s in results["p1"]]
        assert len(set(sample_ids)) == 5

    async def test_factory_receives_action(self, tmp_path):
        """Factory receives the action for per-sample configuration."""
        received_actions = []

        async def factory(action):
            received_actions.append(action)
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = AsyncMock(return_value=StepResult(observation=Observation()))
            env.cleanup = AsyncMock()
            return env

        actions = [Action(message="q1"), Action(message="q2")]
        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        await evaluator.run(actions)

        assert len(received_actions) == 2
        assert received_actions[0].message == "q1"
        assert received_actions[1].message == "q2"

    async def test_max_concurrency(self, tmp_path):
        """max_concurrency limits concurrent env calls."""
        import asyncio

        concurrent_count = 0
        max_concurrent = 0

        async def mock_step(action):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return StepResult(observation=Observation())

        async def factory(action):
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = mock_step
            env.cleanup = AsyncMock()
            return env

        actions = [Action(message=f"q{i}") for i in range(10)]

        evaluator = Evaluator(env_factory=factory, max_concurrency=3, output_path=tmp_path / "results.jsonl")
        await evaluator.run(actions)

        assert max_concurrent <= 3

    async def test_empty_actions(self, mock_env, tmp_path):
        """Empty actions produces empty results."""

        async def factory(action):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([])

        assert results == {}
        mock_env.step.assert_not_awaited()


# ---------------------------------------------------------------------------
# Checkpoint/Resume
# ---------------------------------------------------------------------------


class TestCheckpoint:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock(spec=Environment)
        env.reset = AsyncMock()
        env.step = AsyncMock()
        env.cleanup = AsyncMock()
        return env

    async def test_saves_checkpoint(self, mock_env, tmp_path):
        """Results saved to output_path."""
        mock_env.step.return_value = StepResult(observation=Observation())
        output_path = tmp_path / "results.jsonl"

        async def factory(action):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator.run([Action(message="q1", task_context=TaskContext(id="s1"))])

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1

    async def test_resumes_from_checkpoint(self, mock_env, tmp_path):
        """Skips already-completed samples on resume."""
        mock_env.step.return_value = StepResult(observation=Observation())
        output_path = tmp_path / "results.jsonl"

        async def factory(action):
            return mock_env

        # First run - complete s1
        evaluator1 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator1.run([Action(message="q1", task_context=TaskContext(id="s1"))])
        assert mock_env.step.await_count == 1

        # Second run - s1 skipped, s2 processed
        mock_env.step.reset_mock()
        evaluator2 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results = await evaluator2.run(
            [
                Action(message="q1", task_context=TaskContext(id="s1")),
                Action(message="q2", task_context=TaskContext(id="s2")),
            ]
        )

        assert mock_env.step.await_count == 1  # Only s2 was processed
        assert len(results) == 2  # Both prompt_ids in results
        assert sum(len(samples) for samples in results.values()) == 2


# ---------------------------------------------------------------------------
# pass@k metric
# ---------------------------------------------------------------------------


class TestPassAtKMetric:
    def _make_sample(self, reward: float, idx: int = 0) -> EvalSample:
        return EvalSample(
            action=Action(message="q", task_context=TaskContext(id=f"sample_{idx}")),
            step_result=StepResult(observation=Observation(), reward=RewardResult(reward=reward)),
        )

    def test_empty_samples(self):
        result = compute_pass_at_k({}, k_values=[1])
        assert result == {"pass@1": 0.0}

    def test_all_correct(self):
        results = {"p1": [self._make_sample(1.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == 1.0

    def test_none_correct(self):
        results = {"p1": [self._make_sample(0.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == 0.0

    def test_multiple_problems(self):
        # p1: 2/2 correct -> pass@1 = 1.0
        # p2: 0/2 correct -> pass@1 = 0.0
        results = {
            "p1": [self._make_sample(1.0, 0), self._make_sample(1.0, 1)],
            "p2": [self._make_sample(0.0, 0), self._make_sample(0.0, 1)],
        }
        # Average: (1.0 + 0.0) / 2 = 0.5
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == pytest.approx(0.5)

    def test_multiple_k_values(self):
        # p1: 1/5 correct
        results = {"p1": [self._make_sample(1.0 if i == 0 else 0.0, i) for i in range(5)]}
        result = compute_pass_at_k(results, k_values=[1, 5])
        # pass@1 = 1 - 4/5 = 0.2
        assert result["pass@1"] == pytest.approx(0.2)
        # pass@5 = 1.0 (n-c=4 < k=5, guaranteed to get the correct one)
        assert result["pass@5"] == pytest.approx(1.0)

    def test_custom_reward_threshold(self):
        results = {"p1": [self._make_sample(0.5, 0)]}
        # Default threshold 1.0 - not correct
        result = compute_pass_at_k(results, k_values=[1], reward_threshold=1.0)
        assert result["pass@1"] == 0.0
        # Threshold 0.5 - correct
        result = compute_pass_at_k(results, k_values=[1], reward_threshold=0.5)
        assert result["pass@1"] == 1.0

    def test_k_larger_than_n_skipped(self):
        # Only 2 samples, k=5 - this problem is skipped
        results = {"p1": [self._make_sample(1.0, 0), self._make_sample(1.0, 1)]}
        result = compute_pass_at_k(results, k_values=[5])
        assert result["pass@5"] == 0.0  # No problems have enough samples

    def test_none_reward_handled(self):
        """Samples with None reward are treated as incorrect."""
        sample = EvalSample(
            action=Action(message="q", task_context=TaskContext(id="p1_0")),
            step_result=StepResult(observation=Observation(), reward=None),
        )
        result = compute_pass_at_k({"p1": [sample]}, k_values=[1])
        assert result["pass@1"] == 0.0

    def test_half_correct(self):
        # pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
        results = {"p1": [self._make_sample(1.0 if i < 5 else 0.0, i) for i in range(10)]}
        result = compute_pass_at_k(results, k_values=[1])
        assert result["pass@1"] == pytest.approx(0.5)

    def test_one_correct_out_of_ten(self):
        results = {"p1": [self._make_sample(1.0 if i == 0 else 0.0, i) for i in range(10)]}
        result = compute_pass_at_k(results, k_values=[1, 5])
        # pass@1 = 1 - 9/10 = 0.1
        assert result["pass@1"] == pytest.approx(0.1)
        # pass@5 = 1 - C(9,5)/C(10,5) = 0.5
        assert result["pass@5"] == pytest.approx(0.5)


class TestComputeMetrics:
    def _make_sample(self, reward: float, idx: int = 0) -> EvalSample:
        return EvalSample(
            action=Action(message="q", task_context=TaskContext(id=f"sample_{idx}")),
            step_result=StepResult(observation=Observation(), reward=RewardResult(reward=reward)),
        )

    async def test_default_metrics(self, tmp_path):
        """Default metric_fns includes pass@k."""

        async def factory(action):
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = AsyncMock(
                return_value=StepResult(
                    observation=Observation(),
                    reward=RewardResult(reward=1.0),
                )
            )
            env.cleanup = AsyncMock()
            return env

        evaluator = Evaluator(env_factory=factory, n_samples_per_prompt=3, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([Action(message="q", task_context=TaskContext(id="p1"))])

        metrics = evaluator.compute_metrics(results)
        assert "pass@1" in metrics
        assert "pass@2" in metrics
        assert "pass@3" in metrics
        assert metrics["pass@1"] == 1.0


# ---------------------------------------------------------------------------
# AIMEEvaluator
# ---------------------------------------------------------------------------


class TestAIME2024Evaluator:
    @pytest.fixture
    def mock_env_factory(self):
        async def factory(action):
            return MagicMock()

        return factory

    def test_benchmark_name(self, mock_env_factory):
        evaluator = AIME2024Evaluator(env_factory=mock_env_factory)
        assert evaluator.benchmark_name == "aime-2024"

    def test_load_dataset_mocked(self, mock_env_factory, mocker):
        """load_dataset returns Action objects from HuggingFace dataset."""
        mock_dataset = [
            {"id": 1, "problem": "What is 1+1?", "answer": "2"},
            {"id": 2, "problem": "What is 2+2?", "answer": "4"},
        ]
        mocker.patch("strands_env.eval.benchmarks.aime.load_dataset", return_value=mock_dataset)

        evaluator = AIME2024Evaluator(env_factory=mock_env_factory)
        actions = list(evaluator.load_dataset())

        assert len(actions) == 2
        assert actions[0].message == "What is 1+1?"
        assert actions[0].task_context.ground_truth == "2"
        assert actions[0].task_context.id == "aime-2024_1"

    def test_load_dataset_skips_missing_fields(self, mock_env_factory, mocker):
        """Rows with missing problem/answer are skipped."""
        mock_dataset = [
            {"id": 1, "problem": "What is 1+1?", "answer": "2"},
            {"id": 2, "problem": None, "answer": "4"},  # Missing problem
            {"id": 3, "problem": "What is 3+3?", "answer": None},  # Missing answer
        ]
        mocker.patch("strands_env.eval.benchmarks.aime.load_dataset", return_value=mock_dataset)

        evaluator = AIME2024Evaluator(env_factory=mock_env_factory)
        actions = list(evaluator.load_dataset())

        assert len(actions) == 1
        assert actions[0].task_context.id == "aime-2024_1"

    def test_load_dataset_uses_index_for_missing_id(self, mock_env_factory, mocker):
        """Falls back to row index if id field is missing."""
        mock_dataset = [
            {"problem": "What is 1+1?", "answer": "2"},  # No id field
        ]
        mocker.patch("strands_env.eval.benchmarks.aime.load_dataset", return_value=mock_dataset)

        evaluator = AIME2024Evaluator(env_factory=mock_env_factory)
        actions = list(evaluator.load_dataset())

        assert len(actions) == 1
        assert actions[0].task_context.id == "aime-2024_0"  # Uses index 0


# ---------------------------------------------------------------------------
# validate_sample / aborted
# ---------------------------------------------------------------------------


class TestValidateSample:
    def _make_sample(self, reward: float, idx: int = 0, aborted: bool = False) -> EvalSample:
        return EvalSample(
            action=Action(message="q", task_context=TaskContext(id=f"sample_{idx}")),
            step_result=StepResult(observation=Observation(), reward=RewardResult(reward=reward)),
            aborted=aborted,
        )

    def test_default_returns_true(self):
        """Default validate_sample always returns True."""

        async def factory(action):
            return MagicMock()

        evaluator = Evaluator(env_factory=factory)
        sample = self._make_sample(1.0)
        assert evaluator.validate_sample(sample) is True

    async def test_aborted_samples_excluded_from_metrics(self, tmp_path):
        """Entire prompt is excluded from metrics if any sample is aborted."""
        results = {
            "p1": [self._make_sample(1.0, 0), self._make_sample(0.0, 1, aborted=True)],
            "p2": [self._make_sample(1.0, 2), self._make_sample(1.0, 3)],
            "p3": [self._make_sample(0.0, 4, aborted=True)],
        }

        async def factory(action):
            return MagicMock()

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        metrics = evaluator.compute_metrics(results, log=False)

        # p1 has one aborted sample -> entire prompt excluded
        # p2 has no aborted samples (2 correct out of 2) -> pass@1 = 1.0
        # p3 entirely aborted -> excluded
        # Only p2 contributes -> pass@1 = 1.0
        assert metrics["pass@1"] == 1.0

    async def test_aborted_samples_retried_on_resume(self, tmp_path):
        """Aborted samples are retried on resume (not added to completed_ids)."""
        step_count = 0

        class AbortingEvaluator(Evaluator):
            def validate_sample(self, sample):
                return False  # Abort everything

        async def factory(action):
            nonlocal step_count
            step_count += 1
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = AsyncMock(return_value=StepResult(observation=Observation()))
            env.cleanup = AsyncMock()
            return env

        output_path = tmp_path / "results.jsonl"

        # First run
        eval1 = AbortingEvaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results1 = await eval1.run([Action(message="q1", task_context=TaskContext(id="s1"))])
        assert step_count == 1
        assert results1["s1"][0].aborted is True

        # Second run - s1 should be retried (aborted samples are not in completed_ids)
        step_count = 0
        eval2 = AbortingEvaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results2 = await eval2.run([Action(message="q1", task_context=TaskContext(id="s1"))])
        assert step_count == 1  # Retried
        assert results2["s1"][0].aborted is True

    async def test_aborted_count_in_log(self, tmp_path, caplog):
        """Skipped prompts and aborted count appear in metric log output."""
        results = {
            "p1": [self._make_sample(1.0, 0), self._make_sample(0.0, 1, aborted=True)],
            "p2": [self._make_sample(1.0, 2)],
        }

        async def factory(action):
            return MagicMock()

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")

        import logging

        with caplog.at_level(logging.INFO):
            evaluator.compute_metrics(results, log=True)

        assert "Skipped 1 prompts due to aborted samples" in caplog.text

    async def test_custom_validate_sample(self, tmp_path):
        """Subclass can override validate_sample to abort specific samples."""

        class RewardCheckEvaluator(Evaluator):
            def validate_sample(self, sample):
                return sample.step_result.reward is not None

        call_count = 0

        async def factory(action):
            nonlocal call_count
            call_count += 1
            reward = RewardResult(reward=1.0) if call_count == 2 else None
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = AsyncMock(return_value=StepResult(observation=Observation(), reward=reward))
            env.cleanup = AsyncMock()
            return env

        evaluator = RewardCheckEvaluator(
            env_factory=factory,
            n_samples_per_prompt=2,
            max_concurrency=1,
            output_path=tmp_path / "results.jsonl",
        )
        results = await evaluator.run([Action(message="q", task_context=TaskContext(id="p1"))])

        assert len(results["p1"]) == 2
        aborted = [s for s in results["p1"] if s.aborted]
        valid = [s for s in results["p1"] if not s.aborted]
        assert len(aborted) == 1
        assert len(valid) == 1
        assert valid[0].step_result.reward.reward == 1.0
