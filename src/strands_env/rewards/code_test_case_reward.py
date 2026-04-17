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

"""Reward function for algorithmic coding problems with test case validation.

Extracts Python code from agent response, executes it against hidden test cases,
and returns reward as the proportion of test cases passed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult
from strands_env.tools import CodeInterpreterToolkit

logger = logging.getLogger(__name__)

DEFAULT_TEST_CONCURRENCY = 5
DEFAULT_TEST_TIMEOUT = 180


class CodeTestCaseReward(RewardFunction):
    r"""Reward based on proportion of hidden test cases passed.

    Args:
        extract_last_code_block: If `True`, use the last ```python block; else the first.
        test_concurrency: Number of test cases to run in parallel per sample.
        test_timeout: Per-test timeout in seconds. Tests exceeding this are counted as failed.
        region_name: AWS region for bedrock-agentcore.
        role_arn: AWS IAM role ARN for cross-account access.

    Test Case Format:
        `action.task_context.ground_truth` should be:
        ```python
        {
            "inputs": ["test_input_1", "test_input_2", ...],
            "outputs": ["expected_output_1", "expected_output_2", ...],
        }
        ```

    Reward Calculation:
        - `reward = passed / total`, in `[0.0, 1.0]`.
        - Returns `0.0` if code cannot be extracted, ground truth is malformed,
          or the batch of test cases fails with an unexpected exception.
        - Individual execution errors or timeouts count as test failures.

    Notes:
        - Runs test cases in parallel across independent sandbox sessions
          (no cross-test state leakage).
        - Compares outputs with exact string match after stripping whitespace and
          normalizing `\\r\\n` to `\\n`.
        - Call `cleanup()` when done to close sandbox sessions.
    """

    def __init__(
        self,
        extract_last_code_block: bool = True,
        *,
        test_concurrency: int = DEFAULT_TEST_CONCURRENCY,
        test_timeout: int = DEFAULT_TEST_TIMEOUT,
        region_name: str = "us-east-1",
        role_arn: str | None = None,
        max_pool_connections: int = 1024,
        connect_timeout: int = 120,
        read_timeout: int = 120,
        # Backward compat: ignored, aiobotocore manages its own client
        client: Any = None,
    ) -> None:
        """Initialize a `CodeTestCaseReward` instance."""
        self.extract_last_code_block = extract_last_code_block
        self._test_concurrency = test_concurrency
        self._test_timeout = test_timeout
        self._toolkit_kwargs = {
            "region_name": region_name,
            "role_arn": role_arn,
            "max_pool_connections": max_pool_connections,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
        }
        self._toolkits: list[CodeInterpreterToolkit] = [
            CodeInterpreterToolkit(session_name=f"code-reward-{i}", **self._toolkit_kwargs)
            for i in range(test_concurrency)
        ]

    def extract_code(self, text: str) -> str | None:
        """Extract Python code from a markdown code block.

        Args:
            text: Text that may contain ```python ... ``` blocks.

        Returns:
            Extracted code string, or `None` if no code block is found.
        """
        pattern = r"```python\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if not matches:
            pattern = r"```\s*\n(.*?)\n```"
            matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return matches[-1] if self.extract_last_code_block else matches[0]

    async def _execute_one_test(
        self,
        code: str,
        test_input: str,
        expected_output: str,
        idx: int,
        sem: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """Execute a single test case with timeout, using a pooled toolkit."""
        async with sem:
            toolkit = self._toolkits[idx % self._test_concurrency]
            try:
                wrapped = toolkit._wrap_code_with_stdin(code, test_input)
                actual_output = await asyncio.wait_for(
                    toolkit.invoke("executeCode", {"code": wrapped, "language": "python"}),
                    timeout=self._test_timeout,
                )
            except asyncio.TimeoutError:
                actual_output = f"TIMEOUT: exceeded {self._test_timeout}s"
            except Exception as e:  # noqa: BLE001
                actual_output = f"EXECUTION_ERROR: {type(e).__name__}: {str(e)}"

            is_passed = self.compare_outputs(expected_output, actual_output)
            return {
                "test_case": idx,
                "passed": is_passed,
                "expected": expected_output[:200],
                "actual": actual_output[:200],
            }

    def compare_outputs(self, expected: str, actual: str) -> bool:
        """Return `True` if `actual` matches `expected` after whitespace normalization."""
        expected_normalized = expected.strip().replace("\r\n", "\n")
        actual_normalized = actual.strip().replace("\r\n", "\n")
        return expected_normalized == actual_normalized

    async def run_test_cases(
        self,
        code: str,
        test_inputs: list[str],
        test_outputs: list[str],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Run `code` against test cases in parallel with per-test timeout.

        Uses `test_concurrency` independent sandbox sessions. Each test runs in
        its own session — no cross-test state leakage.

        Returns:
            `(passed_count, total_count, per_test_results)`.
        """
        if len(test_inputs) != len(test_outputs):
            logger.error("Mismatched test inputs/outputs lengths: %d vs %d", len(test_inputs), len(test_outputs))
            return 0, len(test_inputs), []

        sem = asyncio.Semaphore(self._test_concurrency)
        tasks = [
            self._execute_one_test(code, inp, out, idx, sem)
            for idx, (inp, out) in enumerate(zip(test_inputs, test_outputs, strict=True))
        ]
        results = await asyncio.gather(*tasks)

        passed = sum(1 for r in results if r["passed"])
        return passed, len(test_inputs), list(results)

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Compute reward based on test case pass rate."""
        content = step_result.observation.final_response
        if content is None:
            return RewardResult(reward=0.0, info={"reason": "no_final_response"})

        code = self.extract_code(content)
        if code is None:
            return RewardResult(
                reward=0.0,
                info={"reason": "no_code_block_found", "response_sample": content[:500]},
            )

        ground_truth = action.task_context.ground_truth
        if not isinstance(ground_truth, dict):
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "invalid_ground_truth_format",
                    "ground_truth_type": type(ground_truth).__name__,
                },
            )

        test_inputs = ground_truth.get("inputs", [])
        test_outputs = ground_truth.get("outputs", [])
        if not test_inputs or not test_outputs:
            return RewardResult(
                reward=0.0,
                info={
                    "reason": "no_test_cases",
                    "inputs_count": len(test_inputs),
                    "outputs_count": len(test_outputs),
                },
            )

        try:
            passed, total, test_results = await self.run_test_cases(code, test_inputs, test_outputs)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to run test cases: %s: %s", type(e).__name__, str(e))
            return RewardResult(
                reward=0.0,
                info={"reason": "test_execution_failed", "error": f"{type(e).__name__}: {str(e)}"},
            )

        reward = passed / total if total > 0 else 0.0
        return RewardResult(
            reward=reward,
            info={
                "passed": passed,
                "total": total,
                "pass_rate": reward,
                "code_extracted": code[:500],
                "test_results": test_results[:5],
            },
        )

    async def cleanup(self) -> None:
        """Release all sandbox sessions."""
        for toolkit in self._toolkits:
            try:
                await toolkit.cleanup()
            except Exception:  # noqa: BLE001
                pass
