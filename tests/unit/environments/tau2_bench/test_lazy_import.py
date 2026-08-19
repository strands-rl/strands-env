from __future__ import annotations

import subprocess
import sys
import textwrap


def test_importing_tau2_bench_does_not_import_tau2() -> None:
    """Importing the env package, shim, and benchmark must not pull in `tau2`."""
    code = textwrap.dedent(
        """
        import sys

        import strands_env.environments.tau2_bench
        from strands_env.environments.tau2_bench import _tau2
        import strands_env.eval.benchmarks.tau2_bench

        leaked = sorted(m for m in sys.modules if m == "tau2" or m.startswith("tau2."))
        assert not leaked, f"tau2 imported at module load: {leaked}"
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
