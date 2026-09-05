from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol

from pydantic import BaseModel, Field, SerializeAsAny
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from strands_env.core import AsyncEnvFactory, RolloutResult
from strands_env.core.types import TaskT

from .metrics import MetricFunction, compute_pass_at_k

if TYPE_CHECKING:
    from strands_env.core.distributed import EnvironmentActorPool

logger = logging.getLogger(__name__)


class EvalSample(BaseModel, Generic[TaskT]):
    """Evaluation sample result."""

    task: SerializeAsAny[TaskT] = Field(..., description="The task that was evaluated.")
    result: RolloutResult = Field(..., description="The rollout result.")
    aborted: bool = Field(default=False, description="Whether this sample was aborted.")


class EvalReporter(Protocol):
    """Broadcasting evaluation results to remote sources."""

    async def publish(self, run_dir: Path) -> None: ...


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
        keep_rollout: bool = False,
        env_actor_pool: EnvironmentActorPool | None = None,
        reporters: Sequence[EvalReporter] = (),
    ):
        """Initialize an `Evaluator` instance.

        Args:
            env_factory: builds a fresh Environment per sample. Required locally, ignored when
                `env_actor_pool` is given.
            max_concurrency: ceiling on concurrent `evaluate_sample()` calls.
            n_samples_per_prompt: samples per prompt; for pass@k set it to `max(k_values)`.
            output_path: JSONL destination, and what makes resume possible — an existing file is
                read back and its completed samples skipped.
            keep_rollout: keep the token-level rollout in results. SGLang backends only; the others
                produce an empty one.
            env_actor_pool: Ray actor pool for distributed evaluation.
            reporters: where `publish()` broadcasts the finished run; each gets the run directory.
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
        self.keep_rollout = keep_rollout
        self.results: dict[str, list[EvalSample[TaskT]]] = defaultdict(list)
        self.completed_ids: set[str] = set()
        self.reporters = list(reporters)

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
        """Load completed samples from checkpoint file; aborted rows are dropped so their retry leaves no duplicate."""
        if not self.output_path.exists():
            return

        self.results = defaultdict(list)
        self.completed_ids = set()

        kept: list[str] = []
        n_aborted = 0
        with open(self.output_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(
                    line, object_hook=lambda d: base64.b64decode(d["__bytes__"]) if set(d) == {"__bytes__"} else d
                )
                sample: EvalSample[TaskT] = EvalSample.model_validate(data)
                if sample.aborted:
                    n_aborted += 1
                    continue
                prompt_id = sample.task.id.rsplit("_", 1)[0]
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample.task.id)
                kept.append(line)
        if n_aborted:
            tmp = self.output_path.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(kept), encoding="utf-8")
            tmp.replace(self.output_path)

        total = sum(len(s) for s in self.results.values())
        aborted_msg = f" (dropped {n_aborted} aborted for retry)" if n_aborted else ""
        logger.info("Resumed %s completed samples%s from %s", total, aborted_msg, self.output_path)

    def log_sample(self, sample: EvalSample[TaskT]) -> None:
        """Append one sample to `results.jsonl` as a readable UTF-8 row; bytes travel as `{"__bytes__": base64}`."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps(
            sample.model_dump(), ensure_ascii=False, default=lambda b: {"__bytes__": base64.b64encode(b).decode()}
        )
        # UTF-8 cannot encode a lone surrogate; backslashreplace writes it as \uXXXX, which JSON reads back.
        with open(self.output_path, "a", encoding="utf-8", errors="backslashreplace") as f:
            f.write(row + "\n")

    async def publish(self) -> None:
        """Broadcast the run directory to every reporter; one that raises is logged and skipped."""
        for reporter in self.reporters:
            try:
                await reporter.publish(self.output_path.parent)
            except Exception:
                logger.exception("reporter %s failed", type(reporter).__name__)

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
            logger.exception("[%s]: evaluate_sample failed, aborting: %s", task.id, e)
            return EvalSample(task=task, result=RolloutResult(), aborted=True)

    async def run(self, tasks: Iterable[TaskT]) -> dict[str, list[EvalSample[TaskT]]]:
        """Run evaluation on tasks with `n_samples_per_prompt` each."""
        self.load_results()

        to_process: list[TaskT] = []
        for task in tasks:
            for i in range(self.n_samples_per_prompt):
                sample_id = f"{task.id}_{i}"
                if sample_id not in self.completed_ids:
                    expanded = task.model_copy(deep=True)
                    expanded.id = sample_id
                    to_process.append(expanded)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        total = len(to_process)

        async def process(task: TaskT, pbar: tqdm) -> None:
            async with semaphore:
                sample = await self.evaluate_sample(task)
                prompt_id = sample.task.id.rsplit("_", 1)[0]
                self.results[prompt_id].append(sample)
                self.completed_ids.add(sample.task.id)
                self.log_sample(sample)
                pbar.update(1)

        with logging_redirect_tqdm():
            with tqdm(total=total, desc=f"Evaluating {self.benchmark_name}", unit="sample", dynamic_ncols=True) as pbar:
                await asyncio.gather(*[process(task, pbar) for task in to_process])

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
