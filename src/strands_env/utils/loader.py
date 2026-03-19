# Copyright 2025-2026 Horizon RL Contributors
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

"""Generic module/function loading utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any


def load_module(name: str) -> ModuleType:
    """Load a Python module from a dotted name (e.g. `my_package.my_module`)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Could not import module '{name}': {e}") from e


def load_function(name: str) -> Callable[..., Any]:
    """Import a callable from a dotted name (e.g. `my_package.my_module.my_func`)."""
    module_path, _, attr_name = name.rpartition(".")
    if not module_path or not attr_name:
        raise ValueError(f"Invalid dotted path (expected 'module.attr'): {name}")

    module = load_module(module_path)

    if not hasattr(module, attr_name):
        raise ValueError(f"Module '{module_path}' has no attribute '{attr_name}'")

    fn = getattr(module, attr_name)
    if not callable(fn):
        raise ValueError(f"'{name}' is not callable")
    return fn
