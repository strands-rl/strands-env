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

"""LLM-as-judge reward function with optional structured output."""

from __future__ import annotations

import itertools
import logging
from abc import abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel
from strands import Agent
from strands.models import Model
from strands.types.exceptions import ModelThrottledException
from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)

#: TypeVar for the judgment model type. Defaults to `BaseModel` when unparameterized.
JudgmentFormat = TypeVar("JudgmentFormat", bound=BaseModel)


class LLMJudgeReward(RewardFunction, Generic[JudgmentFormat]):
    r"""Abstract base for LLM-as-judge reward functions.

    Args:
        judge_model: A single model or a list of models to round-robin across
            (useful for spreading load across AWS profiles to avoid throttling).
        system_prompt: Optional system prompt for the judge.
        default_reward: Reward to return if the judge fails.
        max_model_retries: Max retries on `ModelThrottledException`, cycling
            through the `judge_model` list.  This is complementary to Strands'
            built-in exponential-backoff retry, as outer, model-level retries.

    Notes:
        - Subclasses set `judgment_format` class attribute and implement
          `get_judge_prompt` and `get_reward`.
        - When `judgment_format` is set, uses structured output and passes
          the parsed Pydantic model to `get_reward`. When `None`, passes
          the raw text response instead.

    Example:
        class SimpleQAReward(LLMJudgeReward[SimpleQAJudgment]):
            judgment_format = SimpleQAJudgment

            async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
                return f"Question: {action.message}\\nAnswer: {step_result.observation.final_response}"

            async def get_reward(self, judgment: SimpleQAJudgment | str) -> float:
                return {"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0}[judgment.grade]
    """

    #: Pydantic model for structured output. Subclasses override to enable structured output.
    judgment_format: type[JudgmentFormat] | None = None

    def __init__(
        self,
        judge_model: Model | list[Model],
        *,
        system_prompt: str | None = None,
        default_reward: float = 0.0,
        max_model_retries: int = 1,
    ) -> None:
        """Initialize a `LLMJudgeReward` instance."""
        self.judge_models = itertools.cycle(judge_model if isinstance(judge_model, list) else [judge_model])
        self.system_prompt = system_prompt
        self.default_reward = default_reward
        self.max_model_retries = max_model_retries

    @abstractmethod
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Format the prompt for the judge model."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def get_reward(self, judgment: JudgmentFormat | str) -> float:
        """Get reward from judgment (structured or text)."""
        raise NotImplementedError("Subclasses must implement this method.")

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        try:
            prompt = await self.get_judge_prompt(action, step_result)
        except Exception as e:
            logger.error("Judge prompt rendering failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={"status": "error", "error_type": "prompt_error", "error": str(e)},
            )

        for attempt in range(self.max_model_retries):
            agent = Agent(model=next(self.judge_models), system_prompt=self.system_prompt, tools=[])
            try:
                if self.judgment_format is not None:
                    judgment: JudgmentFormat | str = await agent.structured_output_async(
                        output_model=self.judgment_format, prompt=prompt
                    )
                else:
                    result = await agent.invoke_async(prompt)
                    judgment = result.message.get("content", [{}])[0].get("text", "")
                break
            except Exception as e:
                if isinstance(e, ModelThrottledException) and attempt < self.max_model_retries - 1:
                    logger.warning(
                        "Judge model throttled (attempt %d/%d), retrying", attempt + 1, self.max_model_retries
                    )
                    continue
                logger.error("Judge model invocation failed: %s", e)
                return RewardResult(
                    reward=self.default_reward,
                    info={"status": "error", "error_type": "judge_error", "error": str(e)},
                )

        try:
            reward = await self.get_reward(judgment)
        except Exception as e:
            logger.error("Reward computation for judgment failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={"status": "error", "error_type": "reward_error", "error": str(e)},
            )

        judgment_data = judgment.model_dump() if isinstance(judgment, BaseModel) else judgment
        return RewardResult(reward=reward, info={"status": "success", "judgment": judgment_data})
