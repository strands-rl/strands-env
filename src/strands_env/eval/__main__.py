"""Module entry point for the evaluation CLI: `python -m strands_env.eval`."""

from __future__ import annotations

import os
import sys

from .cli import eval_cmd

# Ensure the current working directory is importable so user-provided hooks resolve.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    eval_cmd()
