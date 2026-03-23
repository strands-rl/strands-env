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

"""Decorator and utility function helpers for `strands_env`."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import wraps
from typing import Any


def requires_env(*env_vars: str) -> Callable[..., Any]:
    """Decorator that validates environment variables at call time.

    Works with both sync and async functions, methods and standalone functions.

    Notes:
        For async tool methods, returns an error string on missing vars.
        For sync functions, raises `EnvironmentError`.

    Example:
        class MyToolkit:
            @tool
            @requires_env("SERPER_API_KEY")
            async def serper_search(self, query: str) -> str:
                api_key = os.environ["SERPER_API_KEY"]
                ...

        @requires_env("MOONSHOT_API_KEY")
        def kimi_model_factory(*, model_id: str = "moonshot/kimi-k2.5") -> ModelFactory:
            ...
    """

    def _check() -> str | None:
        missing = [v for v in env_vars if not os.getenv(v)]
        return f"Error: missing required environment variable(s): {', '.join(missing)}" if missing else None

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if err := _check():
                    return err
                return await fn(*args, **kwargs)

            return async_wrapper

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if err := _check():
                raise OSError(err)
            return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


def cache_by(*key_args: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that caches function results using only the specified arguments as the cache key.

    Arguments not listed in `key_args` are still passed to the function but excluded
    from the cache key, allowing unhashable arguments (dicts, lists, etc.) without
    breaking the cache.

    Args:
        *key_args: Names of the function parameters to include in the cache key.

    Example::

        @cache_by("service_name", "region")
        def get_client(service_name, region="us-east-1", **config_kwargs):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        cache: dict[tuple, Any] = {}
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve positional/keyword args to param names and fill defaults,
            # so e.g. f("s3") and f(service_name="s3") produce the same key.
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple(bound.arguments[k] for k in key_args)
            if key not in cache:
                cache[key] = fn(*args, **kwargs)
            return cache[key]

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator


def with_timeout(timeout: float | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that enforces a timeout on function execution using `ThreadPoolExecutor`.

    This is useful when the function's own timeout mechanism relies on
    `signal.alarm()` (which only works in the main thread). This decorator
    works correctly in all threading contexts.

    Args:
        timeout: Timeout in seconds, or `None` to run without timeout.

    Raises:
        TimeoutError: If the function doesn't complete within `timeout` seconds.

    Example::

        @with_timeout(5)
        def slow_computation():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if timeout is None:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeoutError as e:
                    raise TimeoutError(f"Operation timed out after {timeout} seconds") from e

        return wrapper

    return decorator
