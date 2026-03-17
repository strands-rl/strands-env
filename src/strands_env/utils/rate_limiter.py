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

"""Async rate limiter for API call throttling."""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket rate limiter for asyncio.

    Allows up to `rate` acquisitions per second. Safe for concurrent use from multiple coroutines.

    Uses a reservation strategy: callers that arrive when no tokens are available reserve a
    future token (allowing the bucket to go negative) and sleep the calculated wait time.
    Each caller acquires the lock exactly once and gets a deterministic wait — no retry loop,
    no thundering herd.
    """

    def __init__(self, rate: float):
        """Initialize an `AsyncRateLimiter` instance."""
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self._capacity = int(rate)
        self._tokens = float(self._capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        wait = 0.0
        async with self._lock:
            # 1. Refill tokens based on elapsed time, capped at capacity
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last_refill) * self.rate)
            self._last_refill = now

            # 2. Token available — consume and return immediately
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # 3. No token — reserve a future one (go negative) and calculate wait
            wait = (1.0 - self._tokens) / self.rate
            self._tokens -= 1.0

        # 4. Sleep outside lock so other coroutines can reserve their own slots
        await asyncio.sleep(wait)
