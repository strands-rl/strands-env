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

"""Reward function for algorithmic coding problems with test case validation."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult
from strands_env.tools import CodeInterpreterToolkit
from strands_env.tools.code_interpreter import CodeInterpreterQuotas, create_aio_client

logger = logging.getLogger(__name__)

DEFAULT_TEST_CONCURRENCY = 5
DEFAULT_TEST_TIMEOUT = 180


class CodeTestCaseReward(RewardFunction):
    r"""Reward based on proportion of hidden test cases passed.

    Creates ONE shared aiobotocore client and passes it to all parallel
    `CodeInterpreterToolkit` instances — avoids duplicate STS assume-role
    calls and connection pool exhaustion.
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
        client: Any = None,
    ) -> None:
        """Initialize a `CodeTestCaseReward` instance."""
        self.extract_last_code_block = extract_last_code_block
        self._test_concurrency = test_concurrency
        self._test_timeout = test_timeout
        self._client_kwargs = {
            "region_name": region_name,
            "role_arn": role_arn,
            "max_pool_connections": max_pool_connections,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
        }
        self._shared_client: Any = None
        self._toolkits: list[CodeInterpreterToolkit] | None = None

    async def _ensure_toolkits(self) -> list[CodeInterpreterToolkit]:
        """Lazy-init: create ONE shared client + N toolkits on first use."""
        if self._toolkits is None:
            self._shared_client = await create_aio_client(**self._client_kwargs)
            shared_quotas = CodeInterpreterQuotas()
            self._toolkits = [
                CodeInterpreterToolkit(
                    aio_client=self._shared_client,
                    session_name=f"code-reward-{i}",
                    quotas=shared_quotas,
                )
                for i in range(self._test_concurrency)
            ]
        return self._toolkits

    def extract_code(self, text: str) -> str | None:
        """Extract Python code from a markdown code block."""
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
        toolkits: list[CodeInterpreterToolkit],
    ) -> dict[str, Any]:
        """Execute a single test case with timeout, using a pooled toolkit."""
        async with sem:
            toolkit = toolkits[idx % self._test_concurrency]
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
        return expected.strip().replace("\r\n", "\n") == actual.strip().replace("\r\n", "\n")

    async def run_test_cases(
        self,
        code: str,
        test_inputs: list[str],
        test_outputs: list[str],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        """Run `code` against test cases in parallel with per-test timeout."""
        if len(test_inputs) != len(test_outputs):
            logger.error("Mismatched test inputs/outputs lengths: %d vs %d", len(test_inputs), len(test_outputs))
            return 0, len(test_inputs), []

        toolkits = await self._ensure_toolkits()
        sem = asyncio.Semaphore(self._test_concurrency)
        tasks = [
            self._execute_one_test(code, inp, out, idx, sem, toolkits)
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
                info={"reason": "invalid_ground_truth_format", "ground_truth_type": type(ground_truth).__name__},
            )

        test_inputs = ground_truth.get("inputs", [])
        test_outputs = ground_truth.get("outputs", [])
        if not test_inputs or not test_outputs:
            return RewardResult(
                reward=0.0,
                info={"reason": "no_test_cases", "inputs_count": len(test_inputs), "outputs_count": len(test_outputs)},
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
        """Release all sandbox sessions and the shared client."""
        if self._toolkits:
            for toolkit in self._toolkits:
                try:
                    await toolkit.cleanup()
                except Exception:  # noqa: BLE001
                    pass
        if self._shared_client is not None:
            try:
                await self._shared_client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._shared_client = None
        self._toolkits = None
