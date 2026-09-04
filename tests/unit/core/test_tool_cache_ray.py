from __future__ import annotations

import asyncio
import uuid

import pytest

from strands_env.core.tool_cache import CacheLevel, CacheOutcome

ray = pytest.importorskip("ray")

# `ray.shutdown()` leaves a raylet log handle open; the repo turns that ResourceWarning into an error.
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(scope="module")
def cluster():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, logging_level="ERROR")
    yield
    ray.shutdown()


@ray.remote(num_cpus=0)
class Process:
    """One process of a job: a local cache, the named shared cache, and a slow tool that counts its runs."""

    def __init__(self, cache_name: str, delay: float = 0.3):
        from strands.tools.tools import AgentTool, ToolResultEvent
        from strands.types.tools import ToolResult

        from strands_env.core.tool_cache import CachedTool, ToolCache, get_shared_tool_cache

        class SlowTool(AgentTool):
            runs = 0

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
                SlowTool.runs += 1
                await asyncio.sleep(delay)
                status = "error" if tool_use["input"].get("fail") else "success"
                text = f"run {SlowTool.runs} in {cache_name}"
                yield ToolResultEvent(
                    ToolResult(toolUseId=tool_use["toolUseId"], status=status, content=[{"text": text}])
                )

        self.tool = SlowTool()
        self.cache = ToolCache()
        self.accesses: list[tuple[str, str]] = []
        self.cached = CachedTool(
            self.tool,
            self.cache,
            shared_cache=get_shared_tool_cache(name=cache_name),
            on_access=lambda a: self.accesses.append((a.outcome, a.cache_level)),
        )

    async def call(self, tool_input: dict) -> dict:
        tool_use = {"name": "search", "toolUseId": "t", "input": tool_input}
        events = [e async for e in self.cached.stream(tool_use, invocation_state={})]
        result = events[-1].tool_result
        return {
            "status": result["status"],
            "text": result["content"][0]["text"],
            "runs": self.tool.runs,
            "local": len(self.cache),
        }

    def accesses_seen(self) -> list[tuple[str, str]]:
        return self.accesses


def two_processes():
    name = f"tool_cache_{uuid.uuid4().hex}"  # a fresh actor per test: named actors outlive a test within the session
    return Process.remote(name), Process.remote(name)


@pytest.mark.usefixtures("cluster")
class TestCachedToolAcrossProcesses:
    async def test_concurrent_misses_in_two_processes_run_once(self):
        a, b = two_processes()

        first, second = await asyncio.gather(a.call.remote({"q": "shoes"}), b.call.remote({"q": "shoes"}))

        assert first["text"] == second["text"]
        assert first["runs"] + second["runs"] == 1
        outcomes = sorted(await asyncio.gather(a.accesses_seen.remote(), b.accesses_seen.remote()))
        assert outcomes == [[(CacheOutcome.COALESCED, CacheLevel.JOB)], [(CacheOutcome.MISS, CacheLevel.JOB)]]

    async def test_shared_hit_is_copied_locally(self):
        a, b = two_processes()
        await a.call.remote({"q": "shoes"})

        hit = await b.call.remote({"q": "shoes"})

        assert (hit["runs"], hit["local"]) == (0, 1)
        assert await b.accesses_seen.remote() == [(CacheOutcome.HIT, CacheLevel.JOB)]

    async def test_error_reaches_the_process_waiting_on_it(self):
        a, b = two_processes()

        first, second = await asyncio.gather(a.call.remote({"fail": True}), b.call.remote({"fail": True}))

        assert (first["status"], second["status"]) == ("error", "error")
        assert first["runs"] + second["runs"] == 1
