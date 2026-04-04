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

"""Generic module/function/hook loading utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

if TYPE_CHECKING:
    from strands_env.core import AsyncEnvFactory
    from strands_env.eval import Evaluator

#: Type for the create_env_factory function exported by hook modules.
EnvFactoryCreator: TypeAlias = Callable[..., AsyncEnvFactory]


# ---------------------------------------------------------------------------
# Generic Loading Utilities
# ---------------------------------------------------------------------------


def load_module(name: str) -> ModuleType:
    """Load a Python module from a dotted name (e.g. `my_package.my_module`)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        raise ValueError(f"Could not import module '{name}': {e}") from e


def load_class(name: str) -> type:
    """Import a class from a dotted name (e.g. `my_package.my_module.MyClass`)."""
    module_path, _, class_name = name.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Invalid dotted path (expected 'module.class'): {name}")

    module = load_module(module_path)
    _cls = getattr(module, class_name)
    if not isinstance(_cls, type):
        raise ValueError(f"'{name}' is not a class")
    return _cls


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


# ---------------------------------------------------------------------------
# Hook Loading Utilities
# ---------------------------------------------------------------------------


def load_env_factory_hook(hook_path: str) -> EnvFactoryCreator:
    """Load environment factory hook and return `create_env_factory` function.

    Args:
        hook_path: Dotted path to a module exporting `create_env_factory`.
    """
    try:
        return cast(EnvFactoryCreator, load_function(hook_path + ".create_env_factory"))
    except ValueError as e:
        raise ValueError(f"Could not load environment factory hook from {hook_path}: {e}") from e


def load_evaluator_hook(hook_path: str) -> type[Evaluator]:
    """Load evaluator hook and return the `Evaluator` class.

    Args:
        hook_path: Dotted path to a module exporting `EvaluatorClass`.
    """
    try:
        return cast(type["Evaluator"], load_class(hook_path + ".EvaluatorClass"))
    except ValueError as e:
        raise ValueError(f"Could not load evaluator hook from {hook_path}: {e}") from e
