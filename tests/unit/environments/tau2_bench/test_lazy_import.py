"""Import-hygiene test for the tau2-bench package.

`tau2` reads `TAU2_DATA_DIR` into a frozen module global at import time and
`tau2/__init__.py` eagerly pulls that chain in, so nothing in the package may
import tau2 at module load — every tau2 access is funnelled through `_tau2.py`'s
lazy accessors. This locks the invariant in: a future top-level `import tau2`
(or a hoisted inline import) fails here instead of silently breaking the eval
data-dir ordering. Runs in a subprocess so a clean `sys.modules` is guaranteed
regardless of what the rest of the test session imported.
"""

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
