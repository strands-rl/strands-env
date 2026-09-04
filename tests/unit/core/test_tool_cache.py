from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from strands.tools.tools import AgentTool, ToolResultEvent
from strands.types.tools import ToolResult, ToolUse

from strands_env.core.tool_cache import (
    CachedTool,
    RayToolCache,
    ToolCache,
    ToolCacheAccess,
    ToolCacheActor,
    ToolReservation,
)

MODULE = "strands_env.core.tool_cache"


def result(text: str) -> ToolResult:
    return ToolResult(toolUseId="t", status="success", content=[{"text": text}])


def error(text: str) -> ToolResult:
    return ToolResult(toolUseId="t", status="error", content=[{"text": text}])


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class LocalHandle:
    """Stand-in for a Ray actor handle: `.method.remote(...)` runs the actor in this process."""

    def __init__(self, actor: ToolCacheActor):
        for name in ("reserve", "wait", "insert", "release"):
            setattr(self, name, SimpleNamespace(remote=getattr(actor, name)))


class TestToolCache:
    def test_stores_and_serves(self):
        cache = ToolCache()
        cache["k"] = result("v")

        assert cache.get("k") == result("v")
        assert cache.get("missing") is None

    def test_oversized_result_is_not_stored(self):
        cache = ToolCache(max_entry_bytes=10)

        cache["k"] = result("this is far more than ten bytes")

        assert "k" not in cache

    def test_oversized_insert_evicts_nothing(self):
        cache = ToolCache(max_entries=1, max_entry_bytes=80)
        cache["a"] = result("a")

        cache["b"] = result("this is far more than the per-entry cap allows")

        assert list(cache) == ["a"]

    def test_evicts_least_recently_used_beyond_max_entries(self):
        cache = ToolCache(max_entries=2)
        cache["a"] = result("a")
        cache["b"] = result("b")
        assert cache["a"] == result("a")  # touch: b is now the least recently used
        cache["c"] = result("c")

        assert set(cache) == {"a", "c"}

    def test_evicts_when_bytes_exceed_max_bytes(self):
        one_entry = ToolCache().encoded_size(result("x"))
        cache = ToolCache(max_bytes=one_entry * 2 + 1, max_entry_bytes=one_entry)
        cache["a"] = result("a")
        cache["b"] = result("b")
        cache["c"] = result("c")

        assert set(cache) == {"b", "c"}
        assert cache.currsize <= one_entry * 2 + 1

    def test_expires_after_ttl(self):
        clock = Clock()
        cache = ToolCache(ttl_seconds=100, ttl_jitter=0, timer=clock)
        cache["k"] = result("v")

        clock.now = 99
        assert cache.get("k") == result("v")
        clock.now = 100
        assert cache.get("k") is None

    def test_jitter_varies_the_ttl(self, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.random.uniform", lambda low, high: -0.1)
        clock = Clock()
        cache = ToolCache(ttl_seconds=100, ttl_jitter=0.1, timer=clock)
        cache["k"] = result("v")

        clock.now = 89
        assert cache.get("k") == result("v")
        clock.now = 90
        assert cache.get("k") is None


class TestToolCacheActor:
    async def test_first_caller_runs_and_later_callers_wait(self):
        actor = ToolCacheActor()

        assert await actor.reserve("k") == ToolReservation.RUN
        assert await actor.reserve("k") == ToolReservation.WAIT

    async def test_insert_serves_everyone(self):
        actor = ToolCacheActor()
        await actor.reserve("k")

        await actor.insert("k", result("v"))

        assert await actor.reserve("k") == result("v")

    async def test_error_reaches_its_waiters_and_nobody_after(self):
        actor = ToolCacheActor()
        await actor.reserve("k")
        waiter = asyncio.create_task(actor.wait("k", seconds=5))
        await asyncio.sleep(0)

        await actor.insert("k", error("boom"))

        assert await waiter == error("boom")
        assert await actor.reserve("k") == ToolReservation.RUN  # a latecomer runs it again
        assert len(actor.cache) == 0

    async def test_oversized_success_reaches_its_waiters_and_nobody_after(self):
        actor = ToolCacheActor(ToolCache(max_entry_bytes=10))
        await actor.reserve("k")
        waiter = asyncio.create_task(actor.wait("k", seconds=5))
        await asyncio.sleep(0)

        await actor.insert("k", result("far too large to store"))

        assert await waiter == result("far too large to store")
        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_release_hands_the_key_to_the_next_caller(self):
        actor = ToolCacheActor()
        await actor.reserve("k")

        await actor.release("k")

        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_pending_key_lapses_after_the_lease(self):
        clock = Clock()
        actor = ToolCacheActor(ToolCache(timer=clock), lease_seconds=10)
        assert await actor.reserve("k") == ToolReservation.RUN

        clock.now = 11

        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_evicted_value_does_not_leave_a_stale_pending_key(self):
        actor = ToolCacheActor()
        await actor.reserve("k")
        await actor.insert("k", result("v"))
        del actor.cache["k"]

        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_wait_returns_as_soon_as_the_run_ends(self):
        actor = ToolCacheActor()
        await actor.reserve("k")

        waiter = asyncio.create_task(actor.wait("k", seconds=5))
        await asyncio.sleep(0)
        started = time.perf_counter()
        await actor.insert("k", result("v"))

        assert await waiter == result("v")
        assert time.perf_counter() - started < 0.5

    async def test_wait_returns_nothing_on_release(self):
        actor = ToolCacheActor()
        await actor.reserve("k")

        waiter = asyncio.create_task(actor.wait("k", seconds=5))
        await asyncio.sleep(0)
        await actor.release("k")

        assert await waiter is None
        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_wait_times_out_while_the_run_is_still_going(self):
        actor = ToolCacheActor()
        await actor.reserve("k")

        assert await actor.wait("k", seconds=0.01) is None
        assert await actor.reserve("k") == ToolReservation.WAIT

    async def test_wait_without_a_pending_run_answers_at_once(self):
        actor = ToolCacheActor()
        await actor.insert("k", result("v"))

        assert await actor.wait("k", seconds=5) == result("v")
        assert await actor.wait("other", seconds=5) is None

    async def test_pending_keys_survive_load_beyond_the_cache_capacity(self):
        actor = ToolCacheActor(ToolCache(max_entries=2))
        for key in ("a", "b", "c"):
            assert await actor.reserve(key) == ToolReservation.RUN

        assert await actor.reserve("a") == ToolReservation.WAIT


class CountingTool(AgentTool):
    """Succeeds unless the input says `fail`; counts how often it ran."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    @property
    def tool_name(self):
        return "search"

    @property
    def tool_spec(self):
        return {"name": "search", "description": "d", "inputSchema": {"json": {"type": "object"}}}

    @property
    def tool_type(self):
        return "python"

    async def stream(self, tool_use, invocation_state, **kwargs):
        self.calls += 1
        status = "error" if tool_use["input"].get("fail") else "success"
        yield ToolResultEvent(
            ToolResult(toolUseId=tool_use["toolUseId"], status=status, content=[{"text": f"run {self.calls}"}])
        )


class BlockingTool(CountingTool):
    """Holds every call until `release` is set, so concurrent callers overlap."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, tool_use, invocation_state, **kwargs):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        yield ToolResultEvent(ToolResult(toolUseId=tool_use["toolUseId"], status="success", content=[{"text": "slow"}]))


class RawTool(CountingTool):
    """Non-SDK style: yields the bare result instead of a `ToolResultEvent`."""

    async def stream(self, tool_use, invocation_state, **kwargs):
        self.calls += 1
        yield {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": "raw"}]}


class BrokenTool(CountingTool):
    async def stream(self, tool_use, invocation_state, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover


async def call(tool: AgentTool, tool_input: dict[str, Any], tool_use_id: str = "t") -> ToolResult:
    tool_use: ToolUse = {"name": tool.tool_name, "toolUseId": tool_use_id, "input": tool_input}
    events = [event async for event in tool.stream(tool_use, invocation_state={})]
    assert isinstance(events[-1], ToolResultEvent)
    return events[-1].tool_result


def shared(actor: ToolCacheActor) -> RayToolCache:
    return RayToolCache(LocalHandle(actor))


class TestCachedTool:
    def test_delegates_spec(self):
        inner = CountingTool()
        cached = CachedTool(inner, ToolCache())

        assert (cached.tool_name, cached.tool_spec, cached.tool_type) == ("search", inner.tool_spec, "python")

    async def test_serves_repeat_input_once(self):
        inner = CountingTool()
        cached = CachedTool(inner, ToolCache())

        first = await call(cached, {"q": "shoes"}, "t1")
        second = await call(cached, {"q": "shoes"}, "t2")

        assert inner.calls == 1
        assert first["content"] == second["content"]
        assert (first["toolUseId"], second["toolUseId"]) == ("t1", "t2")

    async def test_key_ignores_input_order(self):
        inner = CountingTool()
        cached = CachedTool(inner, ToolCache())

        await call(cached, {"q": "shoes", "n": 1})
        await call(cached, {"n": 1, "q": "shoes"})

        assert inner.calls == 1

    async def test_different_inputs_run_separately(self):
        inner = CountingTool()
        cached = CachedTool(inner, ToolCache())

        await call(cached, {"q": "shoes"})
        await call(cached, {"q": "boots"})

        assert inner.calls == 2

    async def test_scope_separates_entries(self):
        inner = CountingTool()
        cache = ToolCache()

        await call(CachedTool(inner, cache, scope={"customer_id": "a"}), {"q": "shoes"})
        await call(CachedTool(inner, cache, scope={"customer_id": "b"}), {"q": "shoes"})
        await call(CachedTool(inner, cache, scope={"customer_id": "a"}), {"q": "shoes"})

        assert inner.calls == 2

    async def test_cache_is_shared_across_wrappers(self):
        inner = CountingTool()
        cache = ToolCache()

        await call(CachedTool(inner, cache), {"q": "shoes"})
        await call(CachedTool(inner, cache), {"q": "shoes"})

        assert inner.calls == 1

    async def test_error_results_are_not_kept_locally(self):
        inner = CountingTool()
        cache = ToolCache()
        cached = CachedTool(inner, cache)

        first = await call(cached, {"fail": True})
        await call(cached, {"fail": True})

        assert first["status"] == "error"
        assert inner.calls == 2
        assert len(cache) == 0

    async def test_takes_last_event_of_non_sdk_tool(self):
        cached = CachedTool(RawTool(), ToolCache())

        result_ = await call(cached, {"q": "shoes"}, "t9")

        assert result_ == {"toolUseId": "t9", "status": "success", "content": [{"text": "raw"}]}

    async def test_concurrent_misses_run_once(self):
        inner = BlockingTool()
        cached = CachedTool(inner, ToolCache())

        calls = [asyncio.create_task(call(cached, {"q": "shoes"}, f"t{i}")) for i in range(3)]
        await inner.started.wait()
        inner.release.set()
        results = await asyncio.gather(*calls)

        assert inner.calls == 1
        assert [r["toolUseId"] for r in results] == ["t0", "t1", "t2"]
        assert {r["content"][0]["text"] for r in results} == {"slow"}
        assert CachedTool.inflight == {}

    async def test_cancelled_waiter_leaves_run_intact(self):
        inner = BlockingTool()
        cached = CachedTool(inner, ToolCache())

        owner = asyncio.create_task(call(cached, {"q": "shoes"}, "owner"))
        await inner.started.wait()
        waiter = asyncio.create_task(call(cached, {"q": "shoes"}, "waiter"))
        await asyncio.sleep(0)
        waiter.cancel()
        inner.release.set()

        assert (await owner)["content"] == [{"text": "slow"}]
        await asyncio.wait([waiter])
        assert waiter.cancelled()
        assert inner.calls == 1

    async def test_reports_access_outcomes(self):
        seen: list[tuple[str, str]] = []
        inner = BlockingTool()
        cached = CachedTool(inner, ToolCache(), on_access=lambda a: seen.append((a.outcome, a.cache_level)))

        first = asyncio.create_task(call(cached, {"q": "shoes"}, "t1"))
        await inner.started.wait()
        second = asyncio.create_task(call(cached, {"q": "shoes"}, "t2"))
        await asyncio.sleep(0)
        inner.release.set()
        await asyncio.gather(first, second)
        await call(cached, {"q": "shoes"}, "t3")

        assert seen == [("miss", "process"), ("coalesced", "process"), ("hit", "process")]

    async def test_reports_miss_when_tool_raises(self):
        seen: list[ToolCacheAccess] = []
        cached = CachedTool(BrokenTool(), ToolCache(), on_access=seen.append)

        with pytest.raises(RuntimeError, match="boom"):
            await call(cached, {"q": "shoes"})

        assert [(a.tool_name, a.outcome, a.cache_level) for a in seen] == [("search", "miss", "process")]
        assert seen[0].latency_ms >= 0
        assert seen[0].key == cached.cache_key({"q": "shoes"})
        assert CachedTool.inflight == {}


class Runner:
    """A `run` callable for `get_or_run`: returns `outcome` (a result, or raises it), counting calls."""

    def __init__(self, outcome: ToolResult | Exception = None):
        self.outcome = outcome if outcome is not None else result("ran")
        self.calls = 0

    async def __call__(self) -> ToolResult:
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class TestRayToolCacheGetOrRun:
    async def test_first_caller_runs_and_publishes(self):
        actor = ToolCacheActor()
        run = Runner()

        assert await RayToolCache(LocalHandle(actor)).get_or_run("k", run) == (result("ran"), "miss")
        assert run.calls == 1
        assert await actor.reserve("k") == result("ran")

    async def test_present_result_is_a_hit_without_running(self):
        actor = ToolCacheActor()
        await actor.insert("k", result("v"))
        run = Runner()

        assert await RayToolCache(LocalHandle(actor)).get_or_run("k", run) == (result("v"), "hit")
        assert run.calls == 0

    async def test_waits_for_another_process(self):
        actor = ToolCacheActor()
        assert await actor.reserve("k") == ToolReservation.RUN  # another process owns the run
        run = Runner()

        waiter = asyncio.create_task(RayToolCache(LocalHandle(actor)).get_or_run("k", run))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await actor.insert("k", result("from elsewhere"))

        assert await waiter == (result("from elsewhere"), "coalesced")
        assert run.calls == 0

    async def test_error_is_shared_with_the_waiter_then_run_again(self):
        actor = ToolCacheActor()
        proxy = RayToolCache(LocalHandle(actor))
        assert await actor.reserve("k") == ToolReservation.RUN  # another process owns the run
        waiter = asyncio.create_task(proxy.get_or_run("k", Runner()))
        await asyncio.sleep(0.05)

        await actor.insert("k", error("boom"))
        assert await waiter == (error("boom"), "coalesced")

        run = Runner()
        assert await proxy.get_or_run("k", run) == (result("ran"), "miss")
        assert run.calls == 1

    async def test_raising_run_releases_the_key(self):
        actor = ToolCacheActor()

        with pytest.raises(RuntimeError, match="boom"):
            await RayToolCache(LocalHandle(actor)).get_or_run("k", Runner(RuntimeError("boom")))

        assert await actor.reserve("k") == ToolReservation.RUN

    async def test_unreachable_insert_does_not_fail_the_call(self):
        class Unreachable(ToolCacheActor):
            async def insert(self, key, result):
                raise ConnectionError("actor gone")

        assert await RayToolCache(LocalHandle(Unreachable())).get_or_run("k", Runner()) == (result("ran"), "miss")

    async def test_unreachable_reserve_runs_uncached(self):
        class Unreachable(ToolCacheActor):
            async def reserve(self, key):
                raise ConnectionError("actor gone")

        run = Runner()

        assert await RayToolCache(LocalHandle(Unreachable())).get_or_run("k", run) == (result("ran"), "miss")
        assert run.calls == 1


class TestCachedToolWithSharedCache:
    async def test_shared_hit_is_copied_locally(self):
        actor = ToolCacheActor()
        inner = CountingTool()
        seen: list[tuple[str, str]] = []
        first_local, second_local = ToolCache(), ToolCache()
        first = CachedTool(inner, first_local, shared_cache=shared(actor))
        second = CachedTool(
            inner, second_local, shared_cache=shared(actor), on_access=lambda a: seen.append((a.outcome, a.cache_level))
        )

        await call(first, {"q": "shoes"})
        await call(second, {"q": "shoes"})
        await call(second, {"q": "shoes"})

        assert inner.calls == 1
        assert seen == [("hit", "job"), ("hit", "process")]
        assert (len(first_local), len(second_local)) == (1, 1)

    async def test_owner_reports_a_job_miss_and_publishes(self):
        actor = ToolCacheActor()
        seen: list[tuple[str, str]] = []
        cached = CachedTool(
            CountingTool(),
            ToolCache(),
            shared_cache=shared(actor),
            on_access=lambda a: seen.append((a.outcome, a.cache_level)),
        )

        await call(cached, {"q": "shoes"})

        assert seen == [("miss", "job")]
        assert (await actor.reserve(cached.cache_key({"q": "shoes"})))["status"] == "success"

    async def test_local_coalescer_reports_the_tier_the_run_reached(self):
        seen: list[tuple[str, str]] = []
        inner = BlockingTool()
        cached = CachedTool(
            inner,
            ToolCache(),
            shared_cache=shared(ToolCacheActor()),
            on_access=lambda a: seen.append((a.outcome, a.cache_level)),
        )

        first = asyncio.create_task(call(cached, {"q": "shoes"}, "t1"))
        await inner.started.wait()
        second = asyncio.create_task(call(cached, {"q": "shoes"}, "t2"))
        await asyncio.sleep(0)
        inner.release.set()
        await asyncio.gather(first, second)

        assert seen == [("miss", "job"), ("coalesced", "job")]
