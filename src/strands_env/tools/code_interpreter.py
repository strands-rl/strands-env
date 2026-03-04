# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code sandbox toolkit using AWS Bedrock AgentCore Code Interpreter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from strands import tool

if TYPE_CHECKING:
    from botocore.client import BaseClient

CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"


class CodeInterpreterToolkit:
    """Code toolkit using AWS Bedrock AgentCore Code Interpreter.

    Provides `execute_code` and `execute_command` tools for running Python code
    and shell commands in a sandboxed environment.

    Uses a single shared agentcore session through session ID. Call
    `cleanup` when done to close the session.
    """

    def __init__(
        self,
        client: BaseClient,
        session_name: str = "strands-env",
    ):
        """Initialize the toolkit.

        Args:
            client: boto3 client for bedrock-agentcore service.
            session_name: Name for the code interpreter session.
        """
        self.session_name = session_name
        self._client = client
        self._session_id: str | None = None
        # Adding a session lock here to make sure each CodeInterpreterToolkit only owns one session.
        self._session_lock = asyncio.Lock()

    async def _get_session_id(self) -> str:
        """Get or create a code interpreter session (async, thread-safe)."""
        if self._session_id is None:
            async with self._session_lock:
                # Double-check after acquiring lock
                if self._session_id is not None:  # another coroutine may have set it
                    return self._session_id  # type: ignore[unreachable]

                response = await asyncio.to_thread(
                    self._client.start_code_interpreter_session,
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    name=self.session_name,
                    sessionTimeoutSeconds=3600,
                )
                self._session_id = response["sessionId"]
        return self._session_id

    def _parse_stream_response(self, response: dict[str, Any]) -> str:
        """Parse the EventStream response from invoke_code_interpreter.

        Extracts text content from result events or error messages from exceptions.
        Returns plain text that strands will wrap in tool result format.

        Args:
            response: Raw response from invoke_code_interpreter.

        Returns:
            Text content from execution result or error message.
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
        response = await asyncio.to_thread(
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
        response = await asyncio.to_thread(
            self._client.invoke_code_interpreter,
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=session_id,
            name="executeCommand",
            arguments={"command": command},
        )
        return self._parse_stream_response(response)

    def cleanup(self) -> None:
        """Clean up code interpreter session."""
        if self._session_id:
            try:
                self._client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    sessionId=self._session_id,
                )
            except Exception:
                pass  # Ignore cleanup errors
            self._session_id = None
