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

"""Single chokepoint for importing the optional `tau2` dependency.

`tau2` reads `TAU2_DATA_DIR` into a frozen module global at *import* time
(`tau2/utils/utils.py`), and `tau2/__init__.py` eagerly pulls that chain in — so
tau2 must not be imported until the eval layer has configured the data dir.
Routing every tau2 access through this module keeps the rest of the package
import-pure and confines the workaround to one place. Do not `import tau2`
anywhere else in the package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import RewardType
    from tau2.data_model.tasks import Task as Tau2Task

#: Public surface: classes resolved lazily by `__getattr__`, helpers defined below.
__all__ = ["RewardType", "Tau2Task", "ToolCall", "build_environment", "get_tasks", "user_sim_guidelines"]


def __getattr__(name: str) -> Any:
    """Resolve tau2 classes lazily on first access (PEP 562)."""
    match name:
        case "Tau2Task":
            from tau2.data_model.tasks import Task

            return Task
        case "RewardType":
            from tau2.data_model.tasks import RewardType

            return RewardType
        case "ToolCall":
            from tau2.data_model.message import ToolCall

            return ToolCall
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_environment(domain: str, db: Any = None) -> Any:
    """Construct a fresh tau2 domain `Environment` for `domain`."""
    module = importlib.import_module(f"tau2.domains.{domain}.environment")
    return module.get_environment(db=db)


def get_tasks(domain: str, split: str = "base") -> list[Any]:
    """Return the tau2 task objects for `domain`/`split`."""
    module = importlib.import_module(f"tau2.domains.{domain}.environment")
    return module.get_tasks(task_split_name=split)


def user_sim_guidelines(use_tools: bool) -> str:
    """Return tau2's global user-simulator guidelines."""
    from tau2.user.user_simulator import get_global_user_sim_guidelines

    return get_global_user_sim_guidelines(use_tools=use_tools)
