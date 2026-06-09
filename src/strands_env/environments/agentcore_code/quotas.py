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

"""Shared AWS quotas for the Code Interpreter toolkit."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from aiolimiter import AsyncLimiter


class CodeInterpreterQuotas:
    """Shared AWS quotas for Code Interpreter API operations.

    Notes:
        - Create one instance and pass it to all `CodeInterpreterToolkit` instances
        to enforce account-wide limits across concurrent sessions.
        - Manages three concerns:
            - Session semaphore: caps concurrent sessions (`session_concurrency`).
            - Rate limiters: caps API request initiation rate for start/invoke/stop
              (AWS TPS quotas) to prevent throttling errors.
            - Thread pool executor: sized to match `session_concurrency` so each session can
              have one in-flight blocking boto3 call without starving others.

    References:
        - [AWS Bedrock AgentCore default quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html)
    """

    DEFAULT_SESSION_CONCURRENCY = 1000
    DEFAULT_START_TPS = 30
    DEFAULT_INVOKE_TPS = 30
    DEFAULT_STOP_TPS = 30

    def __init__(
        self,
        session_concurrency: int = DEFAULT_SESSION_CONCURRENCY,
        start_tps: float = DEFAULT_START_TPS,
        invoke_tps: float = DEFAULT_INVOKE_TPS,
        stop_tps: float = DEFAULT_STOP_TPS,
    ):
        """Initialize a `CodeInterpreterQuotas` instance."""
        self.session_semaphore = asyncio.Semaphore(session_concurrency)
        self.start_limiter = AsyncLimiter(start_tps, time_period=1)
        self.invoke_limiter = AsyncLimiter(invoke_tps, time_period=1)
        self.stop_limiter = AsyncLimiter(stop_tps, time_period=1)
        self.executor = ThreadPoolExecutor(max_workers=session_concurrency)

    def to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the quotas thread pool (or default pool if no quotas)."""
        return asyncio.get_running_loop().run_in_executor(self.executor, partial(func, *args, **kwargs))
