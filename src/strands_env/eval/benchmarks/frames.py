from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal, override

from datasets import load_dataset
from pydantic import BaseModel, Field

from strands_env.core import RolloutResult, Task
from strands_env.core.llm_judge_reward import LLMJudgeReward

from ..evaluator import Evaluator
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
    async def get_judge_prompt(self, task: Task, result: RolloutResult) -> str:
        return GRADER_TEMPLATE.format(
            query=task.message,
            ground_truth=task.ground_truth,
            model_response=result.final_response,
        )

    @override
    async def get_reward(self, judgment: FramesJudgment | str) -> float:
        if isinstance(judgment, FramesJudgment):
            return {"TRUE": 1.0, "FALSE": 0.0}[judgment.decision]
        return self.default_reward


@register_eval("frames")
class FramesEvaluator(Evaluator):
    """Evaluator for FRAMES benchmark."""

    hf_dataset_path = "google/frames-benchmark"

    @override
    def load_dataset(self) -> Iterable[Task]:
        """Load FRAMES benchmark dataset from HuggingFace."""
        dataset = load_dataset(self.hf_dataset_path, split="test")

        for i, row in enumerate(dataset):
            prompt, answer = row.get("Prompt"), row.get("Answer")
            if prompt is None or answer is None:
                logger.warning("Row %s: missing Prompt/Answer, skipped", i)
                continue

            yield Task(
                id=f"{self.benchmark_name}_{i}",
                message=str(prompt),
                ground_truth=str(answer),
                # untyped metadata bag, kept in saved results for analysis
                **{"wiki_links": row["wiki_links"], "reasoning_types": row["reasoning_types"]},
            )
