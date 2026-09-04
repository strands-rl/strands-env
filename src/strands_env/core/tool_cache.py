from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import cache, partial
from typing import Any, ClassVar, Final, cast, override

from cachetools import TLRUCache, TTLCache
from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolGenerator, ToolResult, ToolSpec, ToolUse

logger = logging.getLogger(__name__)

LONG_POLL_SECONDS: Final[float] = 1.0  # how long a waiter blocks inside the job cache before re-checking its key


class ToolCache(TLRUCache[str, ToolResult, float]):
    """Tool-result cache: LRU bounded by entries and encoded bytes, with a jittered TTL.

    An entry lives `ttl_seconds`; a result over `max_entry_bytes` is dropped without evicting anything.
    Jitter spreads the expiry of results written in one burst.
    """

    def __init__(
        self,
        *,
        max_entries: int = 3_000,
        max_bytes: int = 128 * 1024 * 1024,
        max_entry_bytes: int = 8 * 1024 * 1024,
        ttl_seconds: float = 3600,
        ttl_jitter: float = 0.1,
        timer: Callable[[], float] = time.monotonic,
    ):
        self.max_entries = max_entries
        self.max_entry_bytes = max_entry_bytes
        self.ttl_seconds = ttl_seconds
        self.ttl_jitter = ttl_jitter
        super().__init__(maxsize=max_bytes, ttu=self.expires_at, timer=timer, getsizeof=self.encoded_size)

    def expires_at(self, key: str, value: ToolResult, now: float) -> float:
        return now + self.ttl_seconds * (1 + random.uniform(-self.ttl_jitter, self.ttl_jitter))

    def encoded_size(self, value: ToolResult) -> int:
        size = len(json.dumps(value, separators=(",", ":")).encode())
        if size > self.max_entry_bytes:
            # cachetools' own signal for a value over `maxsize`, so one `except` in `__setitem__` covers both caps.
            raise ValueError("value too large")
        return size

    @override
    def __setitem__(self, key: str, value: ToolResult) -> None:
        try:
            super().__setitem__(key, value)
        except ValueError:
            return
        while len(self) > self.max_entries:
            self.popitem()


class ToolReservation(Enum):
    """`reserve` outcome when the key has no result yet."""

    RUN = "run"  # the caller owns this key's run and must end it with `insert` or `release`
    WAIT = "wait"  # another caller owns it; poll `reserve` until a result appears


class ToolCacheActor:
    """Job-level tool cache: one Ray actor per job, shared by every process.

    `reserve` gives the first caller for a key the run and tells later callers to wait until the owner
    ends it with `insert` or `release`; a pending key lapses after `lease_seconds`, so a dead owner
    cannot hold it. Only a success is kept in `cache`. Whatever the run produced, an error or a result too
    large to keep included, still reaches the callers waiting on it, and nobody after them.

    Notes:
        Only `wait` awaits, and it touches no state after its await, so every method is atomic on Ray's
        asyncio actor without a lock.
        Pending keys are not fenced: an owner outliving its lease can end the next owner's run early,
        costing at most one duplicate call; stored values are never at risk.
    """

    def __init__(self, cache: ToolCache | None = None, *, lease_seconds: float = 180):
        self.cache = cache if cache is not None else ToolCache()
        # A pending key holds the future its waiters block on, resolved with the run's outcome. Pending keys
        # expire, they are never evicted: an in-flight key must not lose its mark under load.
        self.pending: TTLCache[str, asyncio.Future[ToolResult | None]] = TTLCache(
            maxsize=math.inf, ttl=lease_seconds, timer=self.cache.timer
        )

    async def reserve(self, key: str) -> ToolResult | ToolReservation:
        """Return the result for `key`, or whether the caller runs it or waits for whoever does."""
        if (value := self.cache.get(key)) is not None:
            return value
        if key in self.pending:
            return ToolReservation.WAIT
        self.pending[key] = asyncio.get_running_loop().create_future()
        return ToolReservation.RUN

    async def wait(self, key: str, seconds: float) -> ToolResult | None:
        """Block until the run for `key` ends or `seconds` pass, then return what it produced, if anything."""
        if (future := self.pending.get(key)) is None:
            return self.cache.get(key)
        with contextlib.suppress(TimeoutError):
            # Shielded: a timeout must not cancel the future every other waiter shares.
            return await asyncio.wait_for(asyncio.shield(future), seconds)
        return None

    async def insert(self, key: str, result: ToolResult) -> None:
        """End a run with its result: keep a success in `cache`, and hand the result itself to the waiters."""
        if result["status"] == "success":
            self.cache[key] = result
        if (future := self.pending.pop(key, None)) is not None:
            future.set_result(result)

    async def release(self, key: str) -> None:
        """End a run that produced no result, so the next caller takes the key."""
        if (future := self.pending.pop(key, None)) is not None:
            future.set_result(None)


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"  # this call ran the tool
    COALESCED = "coalesced"  # this call waited on another caller's run


class CacheLevel(StrEnum):
    """The deepest tier a call reached."""

    PROCESS = "process"
    JOB = "job"


class RayToolCache:
    """Client side of the job's shared tool cache: the run-once protocol over a `ToolCacheActor` handle."""

    def __init__(self, actor: Any):
        # An `ActorHandle[ToolCacheActor]`, but Ray types method access on a handle as `Never`, so nothing narrower checks.
        self._actor = actor

    async def _end(self, ending: Awaitable[None]) -> None:
        """Report the run's end to the actor; an actor that cannot be reached must not fail the call."""
        try:
            await ending
        except Exception:
            logger.warning("shared tool cache unreachable; the run's result stays local", exc_info=True)

    async def get_or_run(self, key: str, run: Callable[[], Awaitable[ToolResult]]) -> tuple[ToolResult, CacheOutcome]:
        """Return the job's result for `key`, calling `run` when this process is the one to produce it.

        While another process owns the run, block in the actor until it ends. When the actor cannot be
        reached, `run` anyway: the tier is a cache, and its outage must not fail the call.
        """
        waited = False
        reservation: ToolResult | ToolReservation
        try:
            reservation = await self._actor.reserve.remote(key)
            while reservation is ToolReservation.WAIT:
                waited = True
                value = await self._actor.wait.remote(key, LONG_POLL_SECONDS)
                reservation = value if value is not None else await self._actor.reserve.remote(key)
        except Exception:
            logger.warning("shared tool cache unreachable; running uncached", exc_info=True)
            return await run(), CacheOutcome.MISS
        if not isinstance(reservation, ToolReservation):
            return reservation, CacheOutcome.COALESCED if waited else CacheOutcome.HIT

        # This process owns the run and every other process waits for it to end, with a result or without.
        try:
            result = await run()
        except BaseException:
            await self._end(self._actor.release.remote(key))
            raise
        await self._end(self._actor.insert.remote(key, result))
        return result, CacheOutcome.MISS


@cache
def get_tool_cache() -> ToolCache:
    """Return this process's tool cache: one instance, shared by every environment built in the process.

    A cache built per environment would be dropped with the environment after a single episode.
    """
    return ToolCache()


def get_shared_tool_cache(
    cache: ToolCache | None = None, *, lease_seconds: float = 180, name: str = "tool_cache"
) -> RayToolCache:
    """Return the tool cache shared by every process of this job, creating its actor on the first ask.

    Ray scopes `name` to the job, so every process gets the same actor; the arguments only matter on
    the call that creates it. Keep the returned object for as long as the cache is wanted: Ray reclaims a
    named actor once no process holds a handle to it. Needs `ray` installed.
    """
    import ray

    actor = (
        ray.remote(ToolCacheActor)
        # Waiters block inside `wait`, one asyncio task each, so the actor must admit far more calls than the default 1000.
        .options(name=name, get_if_exists=True, num_cpus=0, max_concurrency=10_000)
        .remote(cache, lease_seconds=lease_seconds)
    )
    return RayToolCache(actor)


@dataclass(frozen=True)
class ToolCacheAccess:
    """One call through a `CachedTool`, reported after it finishes."""

    tool_name: str
    outcome: CacheOutcome
    cache_level: CacheLevel
    latency_ms: float
    key: str


class CachedTool(AgentTool):
    """Serve a tool's result from the cache when the same input was seen before.

    The key covers the tool name, the model's input, and `scope`: whatever else the result depends on
    that the input alone does not carry. A miss runs the wrapped tool once per key at a time, in this
    process and, with `shared_cache`, across the job. A result the shared tier hands back is kept locally
    too, so the next call for it in this process stays in-process. `on_access` gets one `ToolCacheAccess`
    per call.

    Notes:
        Wrap only tools whose result is a function of input and scope; a tool with side effects must
        run every time. Intermediate stream events are not forwarded.
    """

    # One table per process: a run in flight is visible to every wrapper, whichever cache it uses.
    inflight: ClassVar[dict[str, asyncio.Task[tuple[ToolResult, CacheOutcome, CacheLevel]]]] = {}

    def __init__(
        self,
        tool: AgentTool,
        cache: ToolCache,
        *,
        shared_cache: RayToolCache | None = None,
        scope: Mapping[str, Any] | None = None,
        on_access: Callable[[ToolCacheAccess], None] | None = None,
    ):
        super().__init__()
        self._tool = tool
        self.cache = cache
        self.shared_cache = shared_cache
        self.scope = dict(scope or {})
        self.on_access = on_access

    @property
    def tool_name(self) -> str:
        return self._tool.tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        return self._tool.tool_spec

    @property
    def tool_type(self) -> str:
        return self._tool.tool_type

    def cache_key(self, tool_input: Mapping[str, Any]) -> str:
        """Hash of the tool name, `tool_input`, and this wrapper's scope, independent of key order."""
        payload = {"tool": self.tool_name, "input": tool_input, "scope": self.scope}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    async def _run(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolResult:
        """Run the wrapped tool to its result."""
        result: Any = None
        async for event in self._tool.stream(tool_use, invocation_state, **kwargs):
            # An SDK tool ends its stream with a ToolResultEvent; a plain tool's last event is its result.
            result = event.tool_result if isinstance(event, ToolResultEvent) else event
            if isinstance(event, ToolResultEvent):
                break
        return cast(ToolResult, result)

    async def _resolve(
        self, key: str, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResult, CacheOutcome, CacheLevel]:
        """Produce the result for a key this process does not hold: from the shared tier, or by running the tool."""
        run = partial(self._run, tool_use, invocation_state, **kwargs)
        if self.shared_cache is None:
            result, outcome, level = await run(), CacheOutcome.MISS, CacheLevel.PROCESS
        else:
            (result, outcome), level = await self.shared_cache.get_or_run(key, run), CacheLevel.JOB
        if result["status"] == "success":
            self.cache[key] = result  # whether this process ran it or the job had it: the next call stays local
        return result, outcome, level

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Yield the cached result for this input, running the wrapped tool once on a miss."""
        started = time.perf_counter()
        key = self.cache_key(tool_use["input"])
        outcome, level = CacheOutcome.HIT, CacheLevel.PROCESS
        result: ToolResult
        try:
            if (cached := self.cache.get(key)) is not None:  # 1. this process's cache
                result = cached
            elif (task := self.inflight.get(key)) is not None:  # 2. a run already in flight in this process
                outcome = CacheOutcome.COALESCED
                result, _, level = await asyncio.shield(task)  # the tier is the one the run we joined reached
            else:  # 3. the job's shared cache, then the tool itself (`_resolve`)
                # Reported as a miss if the run raises; a finished run reports what it actually was.
                outcome, level = CacheOutcome.MISS, CacheLevel.PROCESS if self.shared_cache is None else CacheLevel.JOB
                task = asyncio.create_task(self._resolve(key, tool_use, invocation_state, **kwargs))
                self.inflight[key] = task
                task.add_done_callback(lambda _: self.inflight.pop(key, None))
                # Shielded so a cancelled waiter does not cancel the run every other waiter depends on.
                result, outcome, level = await asyncio.shield(task)
        finally:
            if self.on_access is not None:
                latency_ms = (time.perf_counter() - started) * 1000
                self.on_access(ToolCacheAccess(self.tool_name, outcome, level, latency_ms, key))
        yield ToolResultEvent(
            ToolResult(toolUseId=tool_use["toolUseId"], status=result["status"], content=result["content"])
        )
