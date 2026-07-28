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

"""Pluggable result sinks for the eval `Evaluator`."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, TextIO

    from strands_env.eval.evaluator import EvalSample

logger = logging.getLogger(__name__)


class EvalReporter(ABC):
    """Base class for eval result reporters.

    Notes:
        The lifecycle (called by `Evaluator`) is as follows:
        1. `log_sample(prompt_id, sample)` — after each completed sample (accumulate, fast)
        2. `flush()` — periodic checkpoint (every save_interval completions + end of run)
        3. `log_metrics(metrics)` — after `compute_metrics()` completes
        4. `log_metadata(metadata)` — run-level params/tags (benchmark, model, backend, ...)
        5. `publish()` — async; do all remote/heavy I/O here (S3 upload, MLflow, etc.)
    """

    @abstractmethod
    def log_sample(self, prompt_id: str, sample: EvalSample[Any]) -> None:
        """Record a completed sample. Must be fast (< 100ms, no remote calls)."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Persist accumulated samples (checkpoint). Called periodically and at end of run."""
        ...

    def rewrite(self, results: dict[str, list[EvalSample[Any]]]) -> None:
        """Reconcile the checkpoint to exactly `results`, dropping stale entries.

        Notes:
            Called once at resume time so retried (previously aborted) samples don't leave a
            duplicate task entry behind. No-op for reporters that don't keep a rewritable local
            checkpoint.
        """
        return None

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record aggregated metrics after `Evaluator.compute_metrics()`."""
        ...

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Record run-level metadata (params, tags, config). Override for remote reporters."""
        return None

    async def publish(self) -> None:
        """Push buffered data to remote destinations. Override for remote reporters."""
        return None


class CompositeReporter(EvalReporter):
    """Fans out to multiple reporters with error isolation.

    Non-fatal contract: if one reporter raises, the others still run and eval continues.
    """

    def __init__(self, reporters: list[EvalReporter]) -> None:
        self.reporters = reporters

    def log_sample(self, prompt_id: str, sample: EvalSample[Any]) -> None:
        """Record the sample on every reporter; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                r.log_sample(prompt_id, sample)
            except Exception:
                logger.warning("Reporter %s.log_sample failed", type(r).__name__, exc_info=True)

    def flush(self) -> None:
        """Checkpoint every reporter; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                r.flush()
            except Exception:
                logger.warning("Reporter %s.flush failed", type(r).__name__, exc_info=True)

    def rewrite(self, results: dict[str, list[EvalSample[Any]]]) -> None:
        """Reconcile every reporter's checkpoint to `results`; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                r.rewrite(results)
            except Exception:
                logger.warning("Reporter %s.rewrite failed", type(r).__name__, exc_info=True)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record the metrics on every reporter; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                r.log_metrics(metrics)
            except Exception:
                logger.warning("Reporter %s.log_metrics failed", type(r).__name__, exc_info=True)

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Record the run metadata on every reporter; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                r.log_metadata(metadata)
            except Exception:
                logger.warning("Reporter %s.log_metadata failed", type(r).__name__, exc_info=True)

    async def publish(self) -> None:
        """Publish every reporter; a failing reporter is logged and skipped."""
        for r in self.reporters:
            try:
                await r.publish()
            except Exception:
                logger.warning("Reporter %s.publish failed", type(r).__name__, exc_info=True)


class LocalReporter(EvalReporter):
    """Writes `results.jsonl` + `metrics.json` (+ `metadata.json`) to a local directory.

    Notes:
        Streams each sample to `results.jsonl` as it completes: `log_sample` is the single
        accumulation point (append) and `flush()` syncs the open file handle to disk. On resume,
        `rewrite()` reconciles the file to the kept samples first, so a retried (previously
        aborted) sample never leaves a duplicate task entry behind.
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self._fh: TextIO | None = None

    def _ensure_open(self) -> TextIO:
        if self._fh is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode preserves samples already on disk from a prior/resumed run.
            self._fh = open(self.output_path, "a", encoding="utf-8")
        return self._fh

    def log_sample(self, prompt_id: str, sample: EvalSample[Any]) -> None:
        """Append the sample to results.jsonl."""
        fh = self._ensure_open()
        data = sample.model_dump()
        data["prompt_id"] = prompt_id
        fh.write(json.dumps(data, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        """Sync the open file handle to disk."""
        if self._fh is not None:
            self._fh.flush()

    def rewrite(self, results: dict[str, list[EvalSample[Any]]]) -> None:
        """Truncate and rewrite `results.jsonl` from the kept samples, purging stale aborted rows.

        Notes:
            Reopens in append mode after so subsequent `log_sample` calls keep streaming.
        """
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for prompt_id, samples in results.items():
                for sample in samples:
                    data = sample.model_dump()
                    data["prompt_id"] = prompt_id
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Write `metrics.json` to the output directory."""
        metrics_path = self.output_path.parent / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Write `metadata.json` to the output directory."""
        metadata_path = self.output_path.parent / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    async def publish(self) -> None:
        """Flush and close the open file handle."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
