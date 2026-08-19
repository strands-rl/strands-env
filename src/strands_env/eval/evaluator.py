from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic

from pydantic import BaseModel, Field, SerializeAsAny
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from strands_env.core import AsyncEnvFactory, RolloutResult
from strands_env.core.types import TaskT

from .metrics import MetricFunction, compute_pass_at_k
from .reporter import EvalReporter, LocalReporter

if TYPE_CHECKING:
    from strands_env.core.distributed import EnvironmentActorPool

logger = logging.getLogger(__name__)


class EvalSample(BaseModel, Generic[TaskT]):
    """Evaluation sample result."""

    task: SerializeAsAny[TaskT] = Field(..., description="The task that was evaluated.")
    result: RolloutResult = Field(..., description="The rollout result.")
    aborted: bool = Field(default=False, description="Whether this sample was aborted.")


class Evaluator(Generic[TaskT]):
    """Evaluator for running concurrent environment evaluations."""

    benchmark_name: ClassVar[str] = ""
    hf_dataset_path: ClassVar[str] = ""  # HuggingFace dataset id, if any
    hf_dataset_config: ClassVar[str] = ""  # HF config/subset within the dataset, if any
    git_url: ClassVar[str] = ""  # git repo the data files are cloned from, if any
    git_ref: ClassVar[str] = ""  # tag/branch/commit pin for git_url ("" = default branch HEAD)

    def __init__(
        self,
        env_factory: AsyncEnvFactory | None = None,
        *,
        max_concurrency: int = 10,
        n_samples_per_prompt: int = 1,
        output_path: Path | str | None = None,
        save_interval: int = 10,
        keep_rollout: bool = False,
        env_actor_pool: EnvironmentActorPool | None = None,
        reporter: EvalReporter | None = None,
    ):
        """Initialize an `Evaluator` instance.

        Args:
            env_factory: builds a fresh Environment per sample. Required locally, ignored when
                `env_actor_pool` is given.
            max_concurrency: ceiling on concurrent `evaluate_sample()` calls.
            n_samples_per_prompt: samples per prompt; for pass@k set it to `max(k_values)`.
            output_path: JSONL destination, and what makes resume possible — an existing file is
                read back and its completed samples skipped.
            save_interval: flush every N completed samples.
            keep_rollout: keep the token-level rollout in results. SGLang backends only; the others
                produce an empty one.
            env_actor_pool: Ray actor pool for distributed evaluation.
            reporter: result sink; defaults to a `LocalReporter` on `output_path`.
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
        self.keep_rollout = keep_rollout

        self.reporter: EvalReporter = reporter if reporter is not None else LocalReporter(self.output_path)

        self.results: dict[str, list[EvalSample[TaskT]]] = defaultdict(list)
        self.completed_ids: set[str] = set()

        # Strands' `recurse_event_loop` adds ~3 frames per tool iteration; the default
        # 1000-frame limit busts at ~iter 321 deep inside the OTel tracer's `json.dumps`.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

    def load_dataset(self) -> Iterable[TaskT]:
        """Load dataset. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement load_dataset()")

    def validate_sample(self, sample: EvalSample[TaskT]) -> bool:
        """Check if a completed sample is valid. Override with benchmark-specific logic.

        `False` marks the sample aborted: excluded from metrics, retried on resume.

        The default rejects anything without a reward that computed — a missing `reward_result`, or
        one whose `info` says `status: "error"`. Metrics count an unscorable sample as *incorrect*
        rather than absent, so keeping it would report a fabricated number.
        """
        reward_result = sample.result.reward_result
        return reward_result is not None and reward_result.info.get("status") != "error"

    def get_metric_fns(self) -> list[MetricFunction]:
        """Return metric functions for evaluation. Override to customize.

        Defaults to pass@k for every k up to `n_samples_per_prompt`.
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
                sample: EvalSample[TaskT] = EvalSample.model_validate(data)
                if sample.aborted:
                    n_aborted += 1
                    continue  # Aborted samples are retried on resume
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample.task.id)

        total = sum(len(s) for s in self.results.values())
        aborted_msg = f" (skipped {n_aborted} aborted for retry)" if n_aborted else ""
        logger.info("Resumed %s completed samples%s from %s", total, aborted_msg, self.output_path)

    def save_results(self) -> None:
        """Checkpoint accumulated samples via the reporter."""
        self.reporter.flush()

    async def evaluate_sample(self, task: TaskT) -> EvalSample[TaskT]:
        """Evaluate a single sample."""
        try:
            if self.env_actor_pool is not None:
                result = await self.env_actor_pool.rollout(task)
            else:
                assert self.env_factory is not None
                env = await self.env_factory()
                result = await env.rollout(task)
            # Dropped by default: a full token trajectory per sample bloats results.jsonl.
            if not self.keep_rollout:
                result.rollout = None
            reward_str = f"{result.reward_result.reward:.2f}" if result.reward_result else "N/A"
            reward_info = result.reward_result.info if result.reward_result else {}
            logger.info(
                "[%s]: terminated=%s | reward=%s | label=%s | reward_info=%s | metrics=%s",
                task.id,
                result.termination_reason.value,
                reward_str,
                task.ground_truth,
                reward_info,
                result.metrics,
            )
            sample = EvalSample(task=task, result=result)
            sample.aborted = not self.validate_sample(sample)
            if sample.aborted:
                logger.warning("[%s]: sample aborted by validate_sample", task.id)
            return sample
        except Exception as e:
            logger.error("[%s]: evaluate_sample failed, aborting: %s", task.id, e)
            return EvalSample(task=task, result=RolloutResult(), aborted=True)

    async def run(self, tasks: Iterable[TaskT]) -> dict[str, list[EvalSample[TaskT]]]:
        """Run evaluation on tasks with `n_samples_per_prompt` each."""
        resumed = self.output_path.exists()
        self.load_results()
        if resumed:
            # Reconcile the checkpoint to the kept samples before streaming new ones, so a retried
            # (previously aborted) sample doesn't leave a stale duplicate task entry on disk.
            self.reporter.rewrite(self.results)

        # Expand tasks to (prompt_id, sample_id, task) tuples
        to_process: list[tuple[str, str, TaskT]] = []
        for task in tasks:
            prompt_id = task.id
            for i in range(self.n_samples_per_prompt):
                sample_id = f"{prompt_id}_{i}"
                if sample_id not in self.completed_ids:
                    expanded = task.model_copy(deep=True)
                    expanded.id = sample_id
                    to_process.append((prompt_id, sample_id, expanded))

        semaphore = asyncio.Semaphore(self.max_concurrency)
        save_counter = 0
        total = len(to_process)

        async def process(prompt_id: str, sample_id: str, task: TaskT, pbar: tqdm) -> None:
            nonlocal save_counter
            async with semaphore:
                sample = await self.evaluate_sample(task)
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample_id)
                self.reporter.log_sample(prompt_id, sample)
                pbar.update(1)
                save_counter += 1
                if save_counter >= self.save_interval:
                    self.save_results()
                    save_counter = 0

        with logging_redirect_tqdm():
            with tqdm(total=total, desc=f"Evaluating {self.benchmark_name}", unit="sample", dynamic_ncols=True) as pbar:
                await asyncio.gather(*[process(pid, sid, a, pbar) for pid, sid, a in to_process])
        self.save_results()

        # A missing reward is deterministic (no reward_fn configured), so samples are kept, not
        # retried — but metrics count them as incorrect, which must not go unnoticed.
        rewardless = sum(
            1 for ss in self.results.values() for s in ss if not s.aborted and s.result.reward_result is None
        )
        if rewardless:
            n = sum(len(ss) for ss in self.results.values())
            logger.warning("%d/%d samples have no reward_result — metrics treat them as incorrect", rewardless, n)
        return dict(self.results)

    def compute_metrics(self, results: dict[str, list[EvalSample]], log: bool = True) -> dict[str, float]:
        """Compute all metrics on results.

        One aborted sample excludes its whole prompt, which keeps n consistent for pass@k.
        """
        filtered = {pid: samples for pid, samples in results.items() if not any(s.aborted for s in samples)}

        metrics = {}
        for fn in self.get_metric_fns():
            metrics.update(fn(filtered))

        if log and metrics:
            n_prompts = len(filtered)
            n_skipped = len(results) - n_prompts
            n_samples = sum(len(s) for s in filtered.values())
            name = self.benchmark_name or "Evaluation"

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
