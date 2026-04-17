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

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter.

Uses aiobotocore for fully async I/O — no threads, no GIL contention.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aiolimiter import AsyncLimiter
from strands import tool

logger = logging.getLogger(__name__)


async def create_aio_client(
    region_name: str = "us-east-1",
    role_arn: str | None = None,
    max_pool_connections: int = 1024,
    connect_timeout: int = 120,
    read_timeout: int = 120,
) -> Any:
    """Create a shared aiobotocore client for bedrock-agentcore.

    Call this ONCE and pass the result to all `CodeInterpreterToolkit` instances.
    """
    import aiobotocore.session
    from botocore.config import Config

    session = aiobotocore.session.get_session()
    config = Config(
        max_pool_connections=max_pool_connections,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )

    kwargs: dict[str, Any] = {
        "service_name": "bedrock-agentcore",
        "region_name": region_name,
        "config": config,
    }

    if role_arn:
        sts_session = aiobotocore.session.get_session()
        async with sts_session.create_client("sts", region_name=region_name) as sts:
            creds = await sts.assume_role(
                RoleArn=role_arn,
                RoleSessionName="strands-env-code-interpreter",
            )
        credentials = creds["Credentials"]
        kwargs["aws_access_key_id"] = credentials["AccessKeyId"]
        kwargs["aws_secret_access_key"] = credentials["SecretAccessKey"]
        kwargs["aws_session_token"] = credentials["SessionToken"]

    return await session.create_client(**kwargs).__aenter__()


class CodeInterpreterQuotas:
    """Shared AWS quotas for Code Interpreter API operations.

    Notes:
        - Create one instance and pass it to all `CodeInterpreterToolkit` instances
        to enforce account-wide limits across concurrent sessions.
        - Manages two concerns:
            - Session semaphore: caps concurrent sessions (`session_concurrency`).
            - Rate limiters: caps API request initiation rate for start/invoke/stop
              (AWS TPS quotas) to prevent throttling errors.

    References:
        - [AWS Bedrock AgentCore default quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html)
    """

    DEFAULT_SESSION_CONCURRENCY = 3000
    DEFAULT_START_TPS = 30
    DEFAULT_INVOKE_TPS = 300
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

    async def _acquire_semaphore_with_warning(self, name: str) -> None:
        """Acquire the session semaphore, logging a warning if it blocked."""
        start = time.time()
        await self.session_semaphore.acquire()
        elapsed = time.time() - start
        if elapsed > 0.001:
            logger.warning(
                "%s semaphore hit limit (max: %d), waited %.3fs",
                name,
                self.session_semaphore._value + 1,
                elapsed,
            )

    async def _acquire_limiter_with_warning(self, limiter: AsyncLimiter, name: str) -> None:
        """Acquire `limiter`, logging a warning if it blocked."""
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        if elapsed > 0.001:
            logger.warning("%s rate limiter hit limit, waited %.3fs", name, elapsed)


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Notes:
        - Provides `execute_code` and `execute_command` tools for running Python code
          and shell commands in a sandboxed environment.
        - Uses aiobotocore for fully async I/O — no threads, no GIL contention.
        - Pass a shared `aio_client` (from `create_aio_client()`) to avoid creating
          duplicate clients and STS assume-role calls.
        - Uses a single shared agentcore session through session ID. Call
          `cleanup` when done to close the session.
    """

    CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

    def __init__(
        self,
        aio_client: Any,
        session_name: str = "strands-env",
        quotas: CodeInterpreterQuotas | None = None,
    ):
        """Initialize a `CodeInterpreterToolkit` instance.

        Args:
            aio_client: Shared aiobotocore client from `create_aio_client()`.
            session_name: Name for the code interpreter session.
            quotas: Shared quotas for rate limiting and session concurrency.
        """
        self.session_name = session_name
        self.session_id: str | None = None
        self.quotas = quotas or CodeInterpreterQuotas()
        self._session_lock = asyncio.Lock()
        self._client = aio_client

    async def start_session(self) -> None:
        """Start a code interpreter session if not already started (async, coroutine-safe)."""
        if self.session_id is None:
            async with self._session_lock:
                if self.session_id is not None:
                    return  # type: ignore[unreachable]

                await self.quotas._acquire_semaphore_with_warning("Session")
                await self.quotas._acquire_limiter_with_warning(self.quotas.start_limiter, "StartSession")
                try:
                    response = await self._client.start_code_interpreter_session(
                        codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                        name=self.session_name,
                        sessionTimeoutSeconds=3600,
                    )
                except Exception:
                    self.quotas.session_semaphore.release()
                    raise
                self.session_id = response["sessionId"]

    @staticmethod
    def _indent_code(code: str, spaces: int) -> str:
        """Indent each non-empty line of `code` by `spaces` spaces."""
        indent = " " * spaces
        lines = code.split("\n")
        return "\n".join(indent + line if line.strip() else line for line in lines)

    @classmethod
    def _wrap_code_with_stdin(cls, code: str, stdin: str) -> str:
        """Wrap user code so `input()` and `sys.stdin` read from `stdin`."""
        escaped_stdin = stdin.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        return f"""import sys
from io import StringIO
import builtins

_input_lines = '''{escaped_stdin}'''.split('\\n')
_input_index = [0]

_original_input = builtins.input
def _mock_input(prompt=''):
    if _input_index[0] >= len(_input_lines):
        raise EOFError('No more input available')
    line = _input_lines[_input_index[0]]
    _input_index[0] += 1
    return line

builtins.input = _mock_input
sys.stdin = StringIO('''{escaped_stdin}''')

try:
{cls._indent_code(code, 4)}
finally:
    builtins.input = _original_input
"""

    async def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke the code interpreter and return parsed response."""
        await self.start_session()
        await self.quotas._acquire_limiter_with_warning(self.quotas.invoke_limiter, "InvokeCodeInterpreter")
        response = await self._client.invoke_code_interpreter(
            codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
            sessionId=self.session_id,
            name=name,
            arguments=arguments,
        )
        stream = response.get("stream")
        if stream is not None:
            async for event in stream:
                if "result" in event:
                    content = event["result"].get("content", [])
                    if isinstance(content, list):
                        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        return "\n".join(texts) if texts else ""
                    return str(content)

                for key in (
                    "accessDeniedException",
                    "conflictException",
                    "internalServerException",
                    "resourceNotFoundException",
                    "serviceQuotaExceededException",
                    "throttlingException",
                    "validationException",
                ):
                    if key in event:
                        return f"{key}: {event[key].get('message', key)}"

        return "No result returned."

    @tool
    async def execute_code(self, code: str) -> str:
        """Execute Python code and return the result."""
        return await self.invoke("executeCode", {"code": code, "language": "python"})

    @tool
    async def execute_code_with_stdin(self, code: str, stdin: str) -> str:
        """Execute Python code with stdin input and return the result."""
        wrapped_code = self._wrap_code_with_stdin(code, stdin)
        return await self.invoke("executeCode", {"code": wrapped_code, "language": "python"})

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command and return the result."""
        return await self.invoke("executeCommand", {"command": command})

    async def cleanup(self) -> None:
        """Clean up code interpreter session (does NOT close the shared client)."""
        if self.session_id:
            await self.quotas._acquire_limiter_with_warning(self.quotas.stop_limiter, "StopSession")
            try:
                await self._client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=self.CODE_INTERPRETER_ID,
                    sessionId=self.session_id,
                )
            except Exception:
                pass
            finally:
                self.quotas.session_semaphore.release()
            self.session_id = None
