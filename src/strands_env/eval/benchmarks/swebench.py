from __future__ import annotations

import subprocess
from pathlib import Path
from typing import override

from strands_env.environments.harbor import SWE_SYSTEM_PROMPT_PATH

from ..registry import register_eval
from .terminal_bench import TerminalBenchEvaluator


@register_eval("swebench-verified")
class SWEBenchVerifiedEvaluator(TerminalBenchEvaluator):
    """Evaluator for SWE-bench Verified.

    Inherits the Terminal-Bench loader (which expects a directory of
    Harbor task subdirectories), and overrides `_download_dataset` to do
    a sparse checkout of the swebench-verified subdir from harbor-datasets.
    """

    git_url = "https://github.com/laude-institute/harbor-datasets.git"
    git_ref = "37db108843a49bb31a592e37a75e2c40dc3f9749"  # test.sh self-exports ~/.local/bin (uv fix)
    git_subdir = "datasets/swebench-verified"
    system_prompt_path: Path | None = SWE_SYSTEM_PROMPT_PATH

    @override
    def _download_dataset(self) -> None:
        """Sparse-checkout the swebench-verified subdir into `self.data_dir`.

        Cloning the full `harbor-datasets` repo is hundreds of MB and most
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
                self.git_url,
                str(repo_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "set", self.git_subdir],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", self.git_ref],
            check=True,
        )

        sub = repo_dir / self.git_subdir
        if not sub.is_dir():
            raise RuntimeError(f"sparse checkout missing {sub}")
        sub.rename(self.data_dir)
        # Best-effort cleanup of the now-empty checkout dir.
        try:
            import shutil

            shutil.rmtree(repo_dir)
        except OSError:
            pass
