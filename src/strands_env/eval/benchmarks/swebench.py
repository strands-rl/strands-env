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

"""Evaluator for SWE-bench Verified (Harbor format).

The Harbor-format SWE-bench Verified tasks live as a subdirectory inside
the larger ``harbor-datasets.git`` repo. We sparse-checkout just that
subdirectory at a pinned commit so we don't have to clone every benchmark.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from typing_extensions import override

from strands_env.core import Action
from strands_env.eval.benchmarks.terminal_bench import TerminalBenchEvaluator

from ..registry import register_eval


@register_eval("swebench-verified")
class SWEBenchVerifiedEvaluator(TerminalBenchEvaluator):
    """Evaluator for SWE-bench Verified.

    Inherits the Terminal-Bench loader (which expects a directory of
    Harbor task subdirectories), and overrides ``_download_dataset`` to do
    a sparse checkout of the swebench-verified subdir from harbor-datasets.
    """

    benchmark_name = "swebench-verified"
    GIT_URL = "https://github.com/laude-institute/harbor-datasets.git"
    GIT_COMMIT = "0d48cdd78e14a1e22afa09abcfc1bf210427d66f"
    SUBDIR = "datasets/swebench-verified"
    data_dir: Path = Path("./data/swebench-verified")

    @override
    def _download_dataset(self) -> None:
        """Sparse-checkout the swebench-verified subdir into ``self.data_dir``.

        Cloning the full ``harbor-datasets`` repo is hundreds of MB and most
        of it is unrelated benchmarks. Sparse checkout fetches only the
        directory we care about.
        """
        self.data_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.data_dir.exists():
            return

        # Check out into a sibling .repo dir, then move the subdir into place.
        repo_dir = self.data_dir.parent / ".harbor-datasets-checkout"
        if repo_dir.exists():
            import shutil

            shutil.rmtree(repo_dir)

        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--depth",
                "1",
                "--branch",
                "main",
                self.GIT_URL,
                str(repo_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", self.SUBDIR],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", self.GIT_COMMIT],
            check=True,
        )

        sub = repo_dir / self.SUBDIR
        if not sub.is_dir():
            raise RuntimeError(f"sparse checkout missing {sub}")
        sub.rename(self.data_dir)
        # Best-effort cleanup of the now-empty checkout dir.
        try:
            import shutil

            shutil.rmtree(repo_dir)
        except OSError:
            pass

    @override
    def load_dataset(self) -> list[Action]:
        """Load swebench-verified Harbor tasks (one Action per task directory)."""
        # Same as the parent, but be explicit about the README/LICENSE filter:
        # the swebench-verified dir contains task subdirs only after the
        # sparse checkout, so the parent's "skip dotfiles" rule is enough.
        return super().load_dataset()
