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

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any

from strands import tool

from strands_env.utils.rate_limiter import AsyncRateLimiter

if TYPE_CHECKING:
    from botocore.client import BaseClient

CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"


class CodeInterpreterQuotas:
    """AWS quotas for Code Interpreter API operations."""

    def __init__(
        self,
        *,
        start: AsyncRateLimiter,
        invoke: AsyncRateLimiter,
        stop: AsyncRateLimiter,
        max_sessions: int = 1000,
    ):
        """Initialize a `CodeInterpreterQuotas` instance.

        Args:
            start: Rate limiter for `StartCodeInterpreterSession` API calls.
            invoke: Rate limiter for `InvokeCodeInterpreter` API calls.
            stop: Rate limiter for `StopCodeInterpreterSession` API calls.
            max_sessions: Max concurrent sessions. Defaults to 1000 (AWS default quota).
        """
        self.start = start
        self.invoke = invoke
        self.stop = stop
        self._max_sessions = max_sessions
        self.session_semaphore = asyncio.Semaphore(max_sessions)
        self.executor = ThreadPoolExecutor(max_workers=max_sessions)

    @classmethod
    def from_defaults(
        cls,
        start_tps: float = 30.0,
        invoke_tps: float = 30.0,
        stop_tps: float = 30.0,
        max_sessions: int = 1000,
    ) -> CodeInterpreterQuotas:
        """Create quotas from AWS default limits.

        Args:
            start_tps: TPS for `StartCodeInterpreterSession`. Defaults to 30 (AWS default quota).
            invoke_tps: TPS for `InvokeCodeInterpreter`. Defaults to 30 (AWS default quota).
            stop_tps: TPS for `StopCodeInterpreterSession`. Defaults to 30 (AWS default quota).
            max_sessions: Max concurrent sessions. Defaults to 1000 (AWS default quota).

        Returns:
            A new `CodeInterpreterQuotas` instance.
        """
        return cls(
            start=AsyncRateLimiter(rate=start_tps),
            invoke=AsyncRateLimiter(rate=invoke_tps),
            stop=AsyncRateLimiter(rate=stop_tps),
            max_sessions=max_sessions,
        )

    def set_max_concurrency(self, avg_invokes_per_session: float, avg_session_duration: float) -> int:
        """Set session semaphore to the max concurrent sessions that invoke TPS can sustain.

        Must be called before any sessions are started; raises `RuntimeError` otherwise.

        Args:
            avg_invokes_per_session: Average number of invoke calls per session.
            avg_session_duration: Average session duration in seconds.

        Returns:
            The derived max concurrency value.

        Raises:
            RuntimeError: If called after sessions have already been started.
        """
        if self.session_semaphore._value != self._max_sessions:
            raise RuntimeError("set_max_concurrency must be called before any sessions are started")

        invoke_limited = int(self.invoke.rate * avg_session_duration / avg_invokes_per_session)
        max_concurrency = min(self._max_sessions, invoke_limited)
        self._max_sessions = max_concurrency
        self.session_semaphore = asyncio.Semaphore(max_concurrency)
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
        return max_concurrency


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        - Provides `execute_code` and `execute_command` tools for running Python code
          and shell commands in a sandboxed environment.
        - Uses a single shared agentcore session through session ID. Call
          `cleanup` when done to close the session.
    """

    def __init__(
        self,
        client: BaseClient,
        session_name: str = "strands-env",
        quotas: CodeInterpreterQuotas | None = None,
    ):
        """Initialize a `CodeInterpreterToolkit` instance.

        Args:
            client: boto3 client for bedrock-agentcore service.
            session_name: Name for the code interpreter session.
            quotas: Shared AWS quotas for rate limiting and session concurrency. Create one
                `CodeInterpreterQuotas` instance and pass it to all toolkit instances
                to enforce account-wide limits.
        """
        self.session_name = session_name
        self._client = client
        self._session_id: str | None = None
        self._quotas = quotas
        self._session_lock = asyncio.Lock()

    async def _to_thread(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the quotas thread pool.

        Args:
            func: The blocking function to run.
            *args: Positional arguments to pass to ``func``.
            **kwargs: Keyword arguments to pass to ``func``.

        Returns:
            The return value of ``func``.
        """
        executor = self._quotas.executor if self._quotas else None
        return await asyncio.get_running_loop().run_in_executor(executor, partial(func, *args, **kwargs))

    async def _get_session_id(self) -> str:
        """Get or create a code interpreter session (async, thread-safe)."""
        if self._session_id is None:
            async with self._session_lock:
                # Double-check after acquiring lock
                if self._session_id is not None:  # another coroutine may have set it
                    return self._session_id  # type: ignore[unreachable]

                if self._quotas:
                    await self._quotas.session_semaphore.acquire()
                    await self._quotas.start.acquire()
                try:
                    response = await self._to_thread(
                        self._client.start_code_interpreter_session,
                        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                        name=self.session_name,
                        sessionTimeoutSeconds=3600,
                    )
                except Exception:
                    # Release semaphore if start fails — session was never created
                    if self._quotas:
                        self._quotas.session_semaphore.release()
                    raise
                self._session_id = response["sessionId"]
        return self._session_id

    def _parse_stream_response(self, response: dict[str, Any]) -> str:
        """Parse the EventStream response from `invoke_code_interpreter`.

        Notes:
            Extracts text content from result events or error messages from exceptions.
            Returns plain text that strands will wrap in tool result format.
        """
        errors: list[str] = []

        for event in response.get("stream", []):
            if "result" in event:
                result = event["result"]
                content = result.get("content", [])
                # Extract text from content list
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts) if texts else str(content)
                return str(content)

            # Check for exception events
            for error_key in (
                "accessDeniedException",
                "conflictException",
                "internalServerException",
                "resourceNotFoundException",
                "serviceQuotaExceededException",
                "throttlingException",
                "validationException",
            ):
                if error_key in event:
                    msg = event[error_key].get("message", error_key)
                    errors.append(f"{error_key}: {msg}")
                    break

        # No result found - return collected errors or generic message
        return "\n".join(errors) if errors else "No result received"

    @tool
    async def execute_code(self, code: str) -> str:
        """Execute Python code and return the result.

        Args:
            code: The Python code to execute.

        Returns:
            Execution output text or error message.
        """
        session_id = await self._get_session_id()
        if self._quotas:
            await self._quotas.invoke.acquire()
        response = await self._to_thread(
            self._client.invoke_code_interpreter,
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=session_id,
            name="executeCode",
            arguments={"code": code, "language": "python"},
        )
        return self._parse_stream_response(response)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command and return the result.

        Args:
            command: The shell command to execute.

        Returns:
            Execution output text or error message.
        """
        session_id = await self._get_session_id()
        if self._quotas:
            await self._quotas.invoke.acquire()
        response = await self._to_thread(
            self._client.invoke_code_interpreter,
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=session_id,
            name="executeCommand",
            arguments={"command": command},
        )
        return self._parse_stream_response(response)

    async def cleanup(self) -> None:
        """Clean up code interpreter session."""
        if self._session_id:
            if self._quotas:
                await self._quotas.stop.acquire()
            try:
                await self._to_thread(
                    self._client.stop_code_interpreter_session,
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    sessionId=self._session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                if self._quotas:
                    self._quotas.session_semaphore.release()
            self._session_id = None
