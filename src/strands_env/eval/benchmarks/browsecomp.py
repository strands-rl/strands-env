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

"""Evaluator for BrowseComp benchmark."""

from __future__ import annotations

import base64
import hashlib
import logging
from collections.abc import Iterable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from typing_extensions import override

from strands_env.core import Action, StepResult, TaskContext
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

from ..evaluator import EvalSample, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {query}

[response]: {model_response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {ground_truth}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
""".strip()


# ---------------------------------------------------------------------------
# Judgment and Reward
# ---------------------------------------------------------------------------


class BrowseCompJudgment(BaseModel):
    """Judgment for BrowseComp benchmark."""

    correct: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if the extracted answer matches the correct answer, 'no' otherwise.",
    )


class BrowseCompReward(LLMJudgeReward[BrowseCompJudgment]):
    """Reward for BrowseComp benchmark."""

    judgment_format = BrowseCompJudgment

    @override
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Get judge prompt for BrowseComp benchmark."""
        return GRADER_TEMPLATE.format(
            query=action.message,
            ground_truth=action.task_context.ground_truth,
            model_response=step_result.observation.final_response,
        )

    @override
    async def get_reward(self, judgment: BrowseCompJudgment | str) -> float:
        """Get reward for BrowseComp benchmark."""
        if isinstance(judgment, BrowseCompJudgment):
            return {"yes": 1.0, "no": 0.0}[judgment.correct]
        return self.default_reward


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@register_eval("browsecomp")
class BrowseCompEvaluator(Evaluator):
    """Evaluator for BrowseComp benchmark."""

    benchmark_name: str = "browsecomp"

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where the judge failed (e.g. throttling), so they are retried on resume."""
        reward = sample.step_result.reward
        if reward is None:
            return True
        return reward.info.get("status") != "error"

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load BrowseComp dataset from OpenAI's public CSV.

        Problems and answers are XOR-encrypted; decrypted here using the canary column.

        Yields:
            Action objects with decrypted question and ground truth answer.
        """
        df = pd.read_csv(DATASET_URL)

        for i, row in df.iterrows():
            canary = row.get("canary", "")
            problem = row.get("problem", "")
            answer = row.get("answer", "")
            if not canary or not problem or not answer:
                logger.warning("Row %s: missing canary/problem/answer, skipped", i)
                continue

            try:
                decrypted_problem = self.decrypt(str(problem), str(canary))
                decrypted_answer = self.decrypt(str(answer), str(canary))
            except Exception as e:
                logger.warning("Row %s: decryption failed: %s", i, e)
                continue

            yield Action(
                message=decrypted_problem,
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{i}",
                    ground_truth=decrypted_answer,
                ),
            )

    # ---------------------------------------------------------------------------
    # Decryption (XOR with SHA256-derived key, matches OpenAI's simple-evals)
    # ---------------------------------------------------------------------------

    @classmethod
    def derive_key(cls, password: str, length: int) -> bytes:
        """Derive a fixed-length key from the password using SHA256."""
        digest = hashlib.sha256(password.encode()).digest()  # noqa: S324 — not for security; replicates OpenAI simple-evals XOR obfuscation
        return digest * (length // len(digest)) + digest[: length % len(digest)]

    @classmethod
    def decrypt(cls, ciphertext_b64: str, password: str) -> str:
        """Decrypt base64-encoded ciphertext with XOR."""
        encrypted = base64.b64decode(ciphertext_b64)
        key = cls.derive_key(password, len(encrypted))
        return bytes(a ^ b for a, b in zip(encrypted, key, strict=False)).decode()
