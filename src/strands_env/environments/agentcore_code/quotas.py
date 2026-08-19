from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, ClassVar

from aiolimiter import AsyncLimiter


class CodeInterpreterQuotas:
    """Shared AWS quotas for Code Interpreter API operations.

    One instance passed to every `CodeInterpreterToolkit` is what enforces account-wide limits
    across concurrent sessions. It holds three things:

    - a semaphore capping concurrent sessions at `session_concurrency`
    - rate limiters on start/invoke/stop, so AWS TPS quotas produce waiting rather than throttling
      errors
    - a thread pool sized to `session_concurrency`, so every session can hold one blocking boto3
      call in flight without starving the others

    Defaults follow [AWS Bedrock AgentCore quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html).
    """

    DEFAULT_SESSION_CONCURRENCY: ClassVar[int] = 1000
    DEFAULT_START_TPS: ClassVar[int] = 30
    DEFAULT_INVOKE_TPS: ClassVar[int] = 30
    DEFAULT_STOP_TPS: ClassVar[int] = 30

    def __init__(
        self,
        session_concurrency: int = DEFAULT_SESSION_CONCURRENCY,
        start_tps: float = DEFAULT_START_TPS,
        invoke_tps: float = DEFAULT_INVOKE_TPS,
        stop_tps: float = DEFAULT_STOP_TPS,
    ):
        self.session_semaphore = asyncio.Semaphore(session_concurrency)
        self.start_limiter = AsyncLimiter(start_tps, time_period=1)
        self.invoke_limiter = AsyncLimiter(invoke_tps, time_period=1)
        self.stop_limiter = AsyncLimiter(stop_tps, time_period=1)
        self.executor = ThreadPoolExecutor(max_workers=session_concurrency)

    def to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the quotas thread pool."""
        return asyncio.get_running_loop().run_in_executor(self.executor, partial(func, *args, **kwargs))
