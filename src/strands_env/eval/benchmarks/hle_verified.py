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

"""Evaluator for HLE-Verified benchmark (Gold subset)."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from collections.abc import Iterable
from typing import Any, Literal

from datasets import load_dataset
from pydantic import BaseModel, Field
from strands.types.content import Message
from typing_extensions import override

from strands_env.core import Action, StepResult, TaskContext
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

from ..evaluator import EvalSample, Evaluator
from ..registry import register_eval

logger = logging.getLogger(__name__)


# Official HLE grader prompt from https://github.com/centerforaisafety/hle.
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


class HLEJudgment(BaseModel):
    """Judgment for HLE-Verified benchmark."""

    extracted_final_answer: str = Field(
        ...,
        description="The final exact answer extracted from the response, or 'None' if no final answer exists.",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of whether the extracted answer matches the correct answer.",
    )
    correct: Literal["yes", "no"] = Field(
        ...,
        description="'yes' if extracted_final_answer matches the correct answer, 'no' otherwise.",
    )


class HLEReward(LLMJudgeReward[HLEJudgment]):
    """Reward for HLE-Verified benchmark."""

    judgment_format = HLEJudgment

    @override
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Get judge prompt for HLE-Verified benchmark."""
        return GRADER_TEMPLATE.format(
            query=action.message,
            ground_truth=action.task_context.ground_truth,
            model_response=step_result.observation.final_response,
        )

    @override
    async def get_reward(self, judgment: HLEJudgment | str) -> float:
        """Get reward for HLE-Verified benchmark."""
        if isinstance(judgment, HLEJudgment):
            return {"yes": 1.0, "no": 0.0}[judgment.correct]
        return self.default_reward


class HLEVerifiedEvaluator(Evaluator):
    """Base evaluator for the HLE-Verified Gold subset.

    Loads the `skylenage-ai/HLE-Verified` dataset and keeps only samples whose
    `Verified_Classes` field is `"Gold subset"` (668 fully validated items).
    Subclasses may set `text_only=True` to further drop samples whose nested
    `json.image` field is non-empty (i.e. multimodal).
    """

    benchmark_name: str = "hle-verified-gold"
    dataset_path: str = "skylenage-ai/HLE-Verified"
    text_only: bool = False

    _DATA_URL_RE = re.compile(r"^data:image/(?P<fmt>[^;]+);base64,(?P<payload>.*)$", re.IGNORECASE)
    _SUPPORTED_IMAGE_FORMATS = frozenset({"png", "jpeg", "gif", "webp"})

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where the judge failed (e.g. throttling), so they are retried on resume."""
        reward = sample.step_result.reward
        if reward is None:
            return True
        return reward.info.get("status") != "error"

    @staticmethod
    def parse_image_data_url(data_url: str) -> tuple[Literal["png", "jpeg", "gif", "webp"], bytes]:
        """Parse a `data:image/<fmt>;base64,<payload>` URL into (format, bytes).

        Raises:
            ValueError: If the URL is malformed, the format is not one of Strands'
                supported image formats (png/jpeg/gif/webp), or the base64 payload
                cannot be decoded.
        """
        match = HLEVerifiedEvaluator._DATA_URL_RE.match(data_url)
        if not match:
            raise ValueError(f"Not a base64 image data URL: {data_url[:60]!r}")
        fmt = match.group("fmt").lower()
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in HLEVerifiedEvaluator._SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format {fmt!r} (expected one of png/jpeg/gif/webp)")
        try:
            image_bytes = base64.b64decode(match.group("payload"), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"Malformed base64 image payload: {e}") from e
        return fmt, image_bytes  # type: ignore[return-value]

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load the HLE-Verified Gold subset from HuggingFace (streaming).

        Yields:
            Action objects with question text and ground-truth answer. When a
            sample has an inline image, the action message is a multimodal
            `Message` (text + image content block). When `text_only` is set,
            samples whose nested `json.image` field is non-empty are skipped.
        """
        dataset = load_dataset(self.dataset_path, split="train", streaming=True)

        for i, row in enumerate(dataset):
            if row.get("Verified_Classes") != "Gold subset":
                continue

            # The full original record (with image, answer_type, ...) lives under the `json` column.
            raw = row.get("json")
            meta: dict[str, Any] = json.loads(raw) if isinstance(raw, str) else (raw or {})
            image_data_url = meta.get("image") or ""

            if self.text_only and image_data_url:
                continue

            question, answer = row.get("question"), row.get("answer")
            if question is None or answer is None:
                logger.warning("Row %s: missing question/answer, skipped", i)
                continue

            message: str | Message = str(question)
            if image_data_url:
                fmt, image_bytes = self.parse_image_data_url(image_data_url)
                message = {
                    "role": "user",
                    "content": [
                        {"text": str(question)},
                        {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                    ],
                }

            yield Action(
                message=message,
                task_context=TaskContext(
                    id=f"{self.benchmark_name}_{row.get('id', i)}",
                    ground_truth=str(answer),
                    **{
                        "answer_type": meta.get("answer_type"),
                        "raw_subject": row.get("raw_subject"),
                        "category": row.get("category"),
                        "has_image": bool(image_data_url),
                    },
                ),
            )


@register_eval("hle-verified-gold")
class HLEVerifiedGoldEvaluator(HLEVerifiedEvaluator):
    """HLE-Verified Gold subset with 668 fully validated items, includes multimodal."""

    benchmark_name = "hle-verified-gold"
    text_only = False


@register_eval("hle-verified-gold-text")
class HLEVerifiedGoldTextEvaluator(HLEVerifiedEvaluator):
    """HLE-Verified Gold subset with 575 text-only samples."""

    benchmark_name = "hle-verified-gold-text"
    text_only = True
