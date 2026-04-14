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

"""Benchmark registry with auto-discovery.

Evaluator modules are discovered and imported on first registry access.
Benchmarks with missing optional dependencies are skipped with a warning.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import Evaluator

logger = logging.getLogger(__name__)

# Registry: benchmark name -> Evaluator subclass
_BENCHMARKS: dict[str, type[Evaluator]] = {}
# Unavailable modules: module_name -> error message
_UNAVAILABLE: dict[str, str] = {}
_DISCOVERED = False


def register_eval(name: str) -> Callable[[type[Evaluator]], type[Evaluator]]:
    """Decorator to register a benchmark evaluator.

    Example:
        @register_eval("aime-2024")
        class AIME2024Evaluator(Evaluator):
            ...
    """

    def decorator(cls: type[Evaluator]) -> type[Evaluator]:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark '{name}' is already registered")
        _BENCHMARKS[name] = cls
        return cls

    return decorator


def _discover_benchmarks() -> None:
    """Discover and import all benchmark modules from the benchmarks/ subdirectory.

    Modules with missing dependencies are tracked as unavailable.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    benchmarks_dir = Path(__file__).parent / "benchmarks"

    for py_file in benchmarks_dir.glob("*.py"):
        module_name = py_file.stem
        if module_name.startswith("_"):
            continue

        try:
            importlib.import_module(f"strands_env.eval.benchmarks.{module_name}")
        except ImportError as e:
            _UNAVAILABLE[module_name] = str(e)
            logger.debug("Skipping benchmark module '%s': %s", module_name, e)

    _DISCOVERED = True


def get_benchmark(name: str) -> type[Evaluator]:
    """Get a registered benchmark evaluator by name.

    Args:
        name: Benchmark name (e.g., `"aime-2024"`).
    """
    _discover_benchmarks()

    if name not in _BENCHMARKS:
        available = ", ".join(sorted(_BENCHMARKS.keys())) or "(none)"
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return _BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    """List all registered benchmark names.

    Notes:
        Benchmarks with missing dependencies will not appear in this list.
        Use `list_unavailable_benchmarks()` to see them.
    """
    _discover_benchmarks()
    return sorted(_BENCHMARKS.keys())


def list_unavailable_benchmarks() -> dict[str, str]:
    """Return benchmark modules that failed to load, mapped to their error messages."""
    _discover_benchmarks()
    return dict(_UNAVAILABLE)
