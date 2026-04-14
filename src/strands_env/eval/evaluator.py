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

"""Evaluator for running agentic benchmarks with Strands Agents Environments."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from strands_env.core import Action, AsyncEnvFactory, Observation, StepResult

from .metrics import MetricFn, compute_pass_at_k

if TYPE_CHECKING:
    from strands_env.utils.ray import EnvironmentActorPool

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """Evaluation sample result."""

    action: Action
    """The action (task) that was evaluated."""

    step_result: StepResult
    """The result of the step (observation, reward, termination reason)."""

    aborted: bool = False
    """Whether this sample was aborted (excluded from metrics, retried on resume)."""


class Evaluator:
    """Evaluator for running concurrent environment evaluations."""

    benchmark_name: str = ""
    """Benchmark identifier. Override in subclasses."""

    def __init__(
        self,
        env_factory: AsyncEnvFactory | None = None,
        *,
        max_concurrency: int = 10,
        n_samples_per_prompt: int = 1,
        output_path: Path | str | None = None,
        save_interval: int = 10,
        keep_tokens: bool = False,
        env_actor_pool: EnvironmentActorPool | None = None,
    ):
        """Initialize an `Evaluator` instance.

        Args:
            env_factory: Async factory function that creates a fresh Environment per sample.
                Required for local evaluation; unused when `env_actor_pool` is provided.
            max_concurrency: Maximum concurrent evaluate_sample() calls.
            n_samples_per_prompt: Number of samples per prompt (for pass@k, set to max(k_values)).
            output_path: Path to JSONL file for saving results. Enables resume.
            save_interval: Flush results to disk every N completed samples.
            keep_tokens: Keep token-level observation in results (only valid for `SGLangModel` backends).
            env_actor_pool: Optional Ray actor pool for distributed evaluation.
        """
        if env_factory is None and env_actor_pool is None:
            raise ValueError("Must provide either env_factory or env_actor_pool")
        if output_path is None:
            output_path = Path.cwd() / "results.jsonl"
        self.env_factory = env_factory
        self.env_actor_pool = env_actor_pool
        self.max_concurrency = max_concurrency
        self.n_samples_per_prompt = n_samples_per_prompt
        self.output_path = Path(output_path)
        self.save_interval = save_interval
        self.keep_tokens = keep_tokens

        # Runtime state
        self.results: dict[str, list[EvalSample]] = defaultdict(list)
        self.completed_ids: set[str] = set()

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement load_dataset()")

    def validate_sample(self, sample: EvalSample) -> bool:
        """Check if a completed sample is valid. Override with benchmark-specific logic.

        Return False to mark the sample as aborted (excluded from metrics, retried on resume).
        """
        return True

    def get_metric_fns(self) -> list[MetricFn]:
        """Return metric functions for evaluation. Override to customize.

        Notes:
            By default, includes pass@k metric based on `n_samples_per_prompt`.
        """
        return [
            partial(
                compute_pass_at_k,
                k_values=list(range(1, self.n_samples_per_prompt + 1)),
                reward_threshold=1.0,
            )
        ]

    def load_results(self) -> None:
        """Load completed samples from checkpoint file."""
        if not self.output_path.exists():
            return

        self.results = defaultdict(list)
        self.completed_ids = set()

        n_aborted = 0
        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data.pop("prompt_id")
                sample = EvalSample.model_validate(data)
                if sample.aborted:
                    n_aborted += 1
                    continue  # Aborted samples are retried on resume
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample.action.task_context.id)

        total = sum(len(s) for s in self.results.values())
        aborted_msg = f" (skipped {n_aborted} aborted for retry)" if n_aborted else ""
        logger.info("Resumed %s completed samples%s from %s", total, aborted_msg, self.output_path)

    def save_results(self) -> None:
        """Save all samples to checkpoint file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for prompt_id, samples in self.results.items():
                for sample in samples:
                    data = sample.model_dump()
                    data["prompt_id"] = prompt_id
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

    async def evaluate_sample(self, action: Action) -> EvalSample:
        """Evaluate a single sample."""
        try:
            # Run evaluation in distributed or local mode
            if self.env_actor_pool is not None:
                step_result = await self.env_actor_pool.step(action)
            else:
                assert self.env_factory is not None
                env = await self.env_factory(action)
                try:
                    await env.reset()
                    step_result = await env.step(action)
                finally:
                    await env.cleanup()
            # Clean up token-level observation if not needed to reduce verbosity
            if not self.keep_tokens:
                step_result.observation.tokens = None
            # Runtime logging for debugging
            reward_str = f"{step_result.reward.reward:.2f}" if step_result.reward else "N/A"
            reward_info = step_result.reward.info if step_result.reward else {}
            logger.info(
                "[%s]: reward=%s | label=%s | reward_info=%s | metrics=%s",
                action.task_context.id,
                reward_str,
                action.task_context.ground_truth,
                reward_info,
                step_result.observation.metrics,
            )
            # Return the evaluation sample with aborted flag set
            sample = EvalSample(action=action, step_result=step_result)
            sample.aborted = not self.validate_sample(sample)
            if sample.aborted:
                logger.warning("[%s]: sample aborted by validate_sample", action.task_context.id)
            return sample
        except Exception as e:
            logger.error("[%s]: evaluate_sample failed, aborting: %s", action.task_context.id, e)
            return EvalSample(action=action, step_result=StepResult(observation=Observation()), aborted=True)

    async def run(self, actions: Iterable[Action]) -> dict[str, list[EvalSample]]:
        """Run evaluation on actions with `n_samples_per_prompt` each."""
        self.load_results()

        # Expand actions to (prompt_id, sample_id, action) tuples
        to_process: list[tuple[str, str, Action]] = []
        for action in actions:
            prompt_id = action.task_context.id
            for i in range(self.n_samples_per_prompt):
                sample_id = f"{prompt_id}_{i}"
                if sample_id not in self.completed_ids:
                    expanded = action.model_copy(deep=True)
                    expanded.task_context.id = sample_id
                    to_process.append((prompt_id, sample_id, expanded))

        semaphore = asyncio.Semaphore(self.max_concurrency)
        save_counter = 0
        total = len(to_process)

        async def process(prompt_id: str, sample_id: str, action: Action, pbar: tqdm) -> None:
            nonlocal save_counter
            async with semaphore:
                sample = await self.evaluate_sample(action)
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample_id)
                pbar.update(1)
                save_counter += 1
                if save_counter >= self.save_interval:
                    self.save_results()
                    save_counter = 0

        with logging_redirect_tqdm():
            with tqdm(total=total, desc=f"Evaluating {self.benchmark_name}", unit="sample", dynamic_ncols=True) as pbar:
                await asyncio.gather(*[process(pid, sid, a, pbar) for pid, sid, a in to_process])
        self.save_results()
        return dict(self.results)

    def compute_metrics(self, results: dict[str, list[EvalSample]], log: bool = True) -> dict[str, float]:
        """Compute all metrics on results.

        Notes:
            Aborted samples are excluded from metric computation.
        """
        # Exclude entire prompt if any sample is aborted (keeps n consistent for pass@k)
        filtered = {pid: samples for pid, samples in results.items() if not any(s.aborted for s in samples)}

        metrics = {}
        for fn in self.get_metric_fns():
            metrics.update(fn(filtered))

        if log and metrics:
            n_prompts = len(filtered)
            n_skipped = len(results) - n_prompts
            n_samples = sum(len(s) for s in filtered.values())
            name = self.benchmark_name or "Evaluation"

            # Build formatted output
            lines = [f"{'─' * 40}", f"  {name} Results", f"{'─' * 40}"]
            lines.append(f"  Prompts: {n_prompts}  Samples (n={self.n_samples_per_prompt}): {n_samples}")
            if n_skipped:
                lines.append(f"  Skipped {n_skipped} prompts due to aborted samples")
            lines.append("")
            for metric, value in sorted(metrics.items()):
                lines.append(f"  {metric:<12} {value:>6.1%}")
            lines.append(f"{'─' * 40}")
            logger.info("\n%s", "\n".join(lines))

        return metrics
