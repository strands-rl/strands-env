from __future__ import annotations

import asyncio
import ctypes
import inspect
import os
import threading
from collections.abc import Callable
from functools import wraps
from typing import Any


def requires_env(*env_vars: str) -> Callable[..., Any]:
    """Decorator that validates environment variables at call time.

    Works on sync and async functions alike, whether methods or standalone. An async function
    returns the error string on a missing var — that is what an agent tool wants, since the message
    reaches the model. A sync function raises `OSError` instead.
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

    Arguments not named in `key_args` are still passed through but excluded from the key, which is
    what lets an unhashable argument (a dict, a list) coexist with caching.

    Example:
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


class TimeoutInterrupt(BaseException):
    """Injected into a timed-out thread to interrupt it.

    Inherits from `BaseException` (not `Exception`) so it escapes most
    library `except Exception` handlers, but unlike `KeyboardInterrupt`
    it won't trigger training-framework shutdown hooks.
    """


def with_timeout(timeout: float | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that enforces a timeout on function execution.

    Runs the wrapped function in a daemon thread and, on timeout, injects `TimeoutInterrupt` into
    it (CPython best-effort) so the abandoned work stops consuming resources. `timeout=None` skips
    the wrapper entirely.

    Use this when the callee's own timeout relies on `signal.alarm()`, which only fires on the main
    thread; this one works from any thread.

    Raises:
        TimeoutError: the function did not finish within `timeout` seconds.

    Example:
        @with_timeout(5)
        def slow_computation():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if timeout is None:
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result: list[Any] = []
            exception: list[BaseException] = []

            def target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except BaseException as e:
                    exception.append(e)

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                if thread.ident is not None:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(thread.ident), ctypes.py_object(TimeoutInterrupt)
                    )
                raise TimeoutError(f"Operation timed out after {timeout} seconds")

            if exception:
                raise exception[0]
            return result[0]

        return wrapper

    return decorator
