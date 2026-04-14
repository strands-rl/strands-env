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

"""Evaluator for FRAMES (Factuality, Retrieval, And reasoning MEasurement Set) benchmark."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel, Field
from typing_extensions import override

from strands_env.core import Action, StepResult, TaskContext
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

from ..evaluator import EvalSample, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


GRADER_TEMPLATE = """
===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {query}
- Predicted Answer: {model_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Decide whether the predicted answer is TRUE or FALSE.
""".strip()


class FramesJudgment(BaseModel):
    """Judgment for FRAMES benchmark."""

    decision: Literal["TRUE", "FALSE"] = Field(
        ...,
        description="TRUE if the ground truth answer is present in the predicted answer, FALSE otherwise.",
    )


class FramesReward(LLMJudgeReward[FramesJudgment]):
    """Reward for FRAMES benchmark."""

    judgment_format = FramesJudgment

    @override
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Get judge prompt for FRAMES benchmark."""
        return GRADER_TEMPLATE.format(
            query=action.message,
            ground_truth=action.task_context.ground_truth,
            model_response=step_result.observation.final_response,
        )

    @override
    async def get_reward(self, judgment: FramesJudgment | str) -> float:
        """Get reward for FRAMES benchmark."""
        if isinstance(judgment, FramesJudgment):
            return {"TRUE": 1.0, "FALSE": 0.0}[judgment.decision]
        return self.default_reward


@register_eval("frames")
class FramesEvaluator(Evaluator):
    """Evaluator for FRAMES benchmark."""

    benchmark_name: str = "frames"
    dataset_path: str = "google/frames-benchmark"

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where the judge failed (e.g. throttling), so they are retried on resume."""
        reward = sample.step_result.reward
        if reward is None:
            return True
        return reward.info.get("status") != "error"

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load FRAMES benchmark dataset from HuggingFace.

        Yields:
            Action objects with question, wiki links in task context, and ground truth.
        """
        dataset = load_dataset(self.dataset_path, split="test")

        for i, row in enumerate(dataset):
            prompt, answer = row.get("Prompt"), row.get("Answer")
            if prompt is None or answer is None:
                logger.warning("Row %s: missing Prompt/Answer, skipped", i)
                continue

            yield Action(
                message=str(prompt),
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{i}",
                    ground_truth=str(answer),
                    **{
                        "wiki_links": row["wiki_links"],
                        "reasoning_types": row["reasoning_types"],
                    },
                ),
            )
