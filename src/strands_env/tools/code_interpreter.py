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
from functools import partial
from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter
from strands import tool

if TYPE_CHECKING:
    from botocore.client import BaseClient

CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

# AWS default quotas for Code Interpreter
DEFAULT_MAX_SESSIONS = 1024
DEFAULT_INVOKE_TPS = 30


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
        concurrency: asyncio.Semaphore | int = DEFAULT_MAX_SESSIONS,
        rate_limiter: AsyncLimiter | None = None,
    ):
        """Initialize a `CodeInterpreterToolkit` instance.

        Args:
            client: boto3 client for bedrock-agentcore service.
            session_name: Name for the code interpreter session.
            concurrency: Semaphore or max concurrent sessions. Defaults to 1024 (AWS default quota).
                Share a single `asyncio.Semaphore` across all toolkit instances to enforce
                account-wide session limits.
            rate_limiter: Rate limiter for `InvokeCodeInterpreter` API calls. Defaults to 30 TPS
                (AWS default quota). Share a single `aiolimiter.AsyncLimiter` across all toolkit
                instances to enforce account-wide TPS limits.
        """
        self.session_name = session_name
        self._client = client
        self._session_id: str | None = None
        self._rate_limiter = rate_limiter or AsyncLimiter(DEFAULT_INVOKE_TPS)
        self._semaphore = concurrency if isinstance(concurrency, asyncio.Semaphore) else asyncio.Semaphore(concurrency)
        self._session_lock = asyncio.Lock()

    async def _to_thread(self, func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a blocking function in the default thread pool.

        Args:
            func: The blocking function to run.
            *args: Positional arguments to pass to `func`.
            **kwargs: Keyword arguments to pass to `func`.

        Returns:
            The return value of `func`.
        """
        return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))

    async def _get_session_id(self) -> str:
        """Get or create a code interpreter session (async, thread-safe)."""
        if self._session_id is None:
            async with self._session_lock:
                # Double-check after acquiring lock
                if self._session_id is not None:  # another coroutine may have set it
                    return self._session_id  # type: ignore[unreachable]

                await self._semaphore.acquire()
                try:
                    response = await self._to_thread(
                        self._client.start_code_interpreter_session,
                        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                        name=self.session_name,
                        sessionTimeoutSeconds=3600,
                    )
                except Exception:
                    # Release semaphore if start fails — session was never created
                    self._semaphore.release()
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
        await self._rate_limiter.acquire()
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
        await self._rate_limiter.acquire()
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
            try:
                await self._to_thread(
                    self._client.stop_code_interpreter_session,
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    sessionId=self._session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self._semaphore.release()
            self._session_id = None
