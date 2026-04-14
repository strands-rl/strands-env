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

"""Unit tests for CodeInterpreterQuotas and CodeInterpreterToolkit."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from strands_env.tools.code_interpreter import CodeInterpreterQuotas, CodeInterpreterToolkit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(session_id: str = "sess-1", invoke_result: str = "42") -> MagicMock:
    """Create a mock boto3 bedrock-agentcore client."""
    client = MagicMock()
    client.start_code_interpreter_session.return_value = {"sessionId": session_id}
    client.invoke_code_interpreter.return_value = {
        "stream": [{"result": {"content": [{"type": "text", "text": invoke_result}]}}]
    }
    client.stop_code_interpreter_session.return_value = {}
    return client


class TestCodeInterpreterToolkit:
    async def test_invoke_and_cleanup(self):
        """Full lifecycle: start session, invoke, parse result, cleanup."""
        client = _mock_client(invoke_result="hello")
        toolkit = CodeInterpreterToolkit(client=client)

        result = await toolkit.invoke("executeCode", {"code": "print('hello')", "language": "python"})
        assert result == "hello"
        assert toolkit.session_id is not None
        client.start_code_interpreter_session.assert_called_once()
        client.invoke_code_interpreter.assert_called_once()

        await toolkit.cleanup()
        assert toolkit.session_id is None
        client.stop_code_interpreter_session.assert_called_once()

    async def test_invoke_parses_error_events(self):
        """Throttling errors in EventStream are returned as strings (issue #24)."""
        client = _mock_client()
        client.invoke_code_interpreter.return_value = {
            "stream": [{"throttlingException": {"message": "Rate exceeded"}}]
        }
        toolkit = CodeInterpreterToolkit(client=client)
        result = await toolkit.invoke("executeCode", {"code": "x", "language": "python"})
        assert "throttlingException" in result
        assert "Rate exceeded" in result

    async def test_invoke_no_result(self):
        client = _mock_client()
        client.invoke_code_interpreter.return_value = {"stream": []}
        toolkit = CodeInterpreterToolkit(client=client)
        result = await toolkit.invoke("executeCode", {"code": "x", "language": "python"})
        assert result == "No result returned."

    async def test_semaphore_released_on_start_failure(self):
        """Semaphore must be released if start_code_interpreter_session fails."""
        client = _mock_client()
        client.start_code_interpreter_session.side_effect = RuntimeError("AWS error")
        quotas = CodeInterpreterQuotas(session_concurrency=1)
        toolkit = CodeInterpreterToolkit(client=client, quotas=quotas)

        with pytest.raises(RuntimeError, match="AWS error"):
            await toolkit.start_session()

        # Semaphore released — another toolkit can still acquire
        assert quotas.session_semaphore._value == 1

    async def test_semaphore_limits_concurrent_sessions(self):
        """With concurrency=1, second toolkit blocks until first cleans up."""
        quotas = CodeInterpreterQuotas(session_concurrency=1)
        tk1 = CodeInterpreterToolkit(client=_mock_client(session_id="s1"), quotas=quotas)
        tk2 = CodeInterpreterToolkit(client=_mock_client(session_id="s2"), quotas=quotas)

        await tk1.start_session()
        assert quotas.session_semaphore._value == 0

        # tk2 should block — verify with timeout
        # Use (TimeoutError, asyncio.TimeoutError) for Python 3.10 compatibility
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(tk2.start_session(), timeout=0.1)

        # After cleanup, tk2 can proceed
        await tk1.cleanup()
        await tk2.start_session()
        assert tk2.session_id == "s2"
        await tk2.cleanup()

    async def test_shared_quotas_across_toolkits(self):
        """Two toolkits sharing quotas respect the same semaphore."""
        quotas = CodeInterpreterQuotas(session_concurrency=2)
        tk1 = CodeInterpreterToolkit(client=_mock_client(session_id="s1"), quotas=quotas)
        tk2 = CodeInterpreterToolkit(client=_mock_client(session_id="s2"), quotas=quotas)

        await tk1.start_session()
        await tk2.start_session()
        assert quotas.session_semaphore._value == 0  # both slots taken

        await tk1.cleanup()
        assert quotas.session_semaphore._value == 1
        await tk2.cleanup()
        assert quotas.session_semaphore._value == 2

    async def test_rate_limiter_throttles_invocations(self):
        """Invoke rate limiter enforces TPS: burst fills capacity, then ~100ms spacing at 10 TPS."""
        quotas = CodeInterpreterQuotas(invoke_tps=10)
        toolkit = CodeInterpreterToolkit(client=_mock_client(), quotas=quotas)

        timestamps = []
        for _ in range(13):
            await toolkit.invoke("executeCode", {"code": "1", "language": "python"})
            timestamps.append(time.monotonic())

        # First 10 calls use burst capacity (~instant), calls 11-13 are rate-limited
        burst_elapsed = (timestamps[9] - timestamps[0]) * 1000
        throttled_elapsed = (timestamps[12] - timestamps[9]) * 1000

        # Burst phase should be fast (< 50ms for 10 calls)
        assert burst_elapsed < 50, f"Burst took {burst_elapsed:.0f}ms, expected < 50ms"
        # Throttled phase: 3 calls at 10 TPS = ~300ms (allow ±50ms tolerance)
        assert throttled_elapsed > 250, f"Throttled phase took {throttled_elapsed:.0f}ms, expected > 250ms"
        assert throttled_elapsed < 400, f"Throttled phase took {throttled_elapsed:.0f}ms, expected < 400ms"

    async def test_semaphore_blocks_then_unblocks(self):
        """Measure that blocked toolkits resume promptly after cleanup."""
        quotas = CodeInterpreterQuotas(session_concurrency=2)
        toolkits = [CodeInterpreterToolkit(client=_mock_client(f"s{i}"), quotas=quotas) for i in range(4)]

        start_times: dict[int, float] = {}
        t0 = time.monotonic()

        async def start_hold_cleanup(idx: int, hold: float) -> None:
            await toolkits[idx].start_session()
            start_times[idx] = time.monotonic() - t0
            await asyncio.sleep(hold)
            await toolkits[idx].cleanup()

        # Toolkits 0,1 get slots immediately and hold for 200ms.
        # Toolkits 2,3 must wait ~200ms for slots to free up.
        await asyncio.gather(
            start_hold_cleanup(0, 0.2),
            start_hold_cleanup(1, 0.2),
            start_hold_cleanup(2, 0.1),
            start_hold_cleanup(3, 0.1),
        )

        # Toolkits 0,1 should start immediately (< 50ms)
        assert start_times[0] < 0.05
        assert start_times[1] < 0.05
        # Toolkits 2,3 should start after ~200ms (when 0,1 release their slots)
        assert start_times[2] > 0.15
        assert start_times[3] > 0.15
