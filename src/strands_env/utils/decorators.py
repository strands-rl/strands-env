# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decorator utilities for `strands_env`."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any


def requires_env(*env_vars: str) -> Callable[..., Any]:
    """Decorator that validates environment variables at call time.

    Returns an error string if any required env var is missing,
    avoiding the need for credential parameters in `__init__`.

    Example::

        class MyToolkit:
            @tool
            @requires_env("SERPER_API_KEY")
            async def serper_search(self, query: str) -> str:
                api_key = os.environ["SERPER_API_KEY"]
                ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            missing = [v for v in env_vars if not os.getenv(v)]
            if missing:
                return f"Error: missing required environment variable(s): {', '.join(missing)}"
            return await fn(self, *args, **kwargs)

        return wrapper

    return decorator
