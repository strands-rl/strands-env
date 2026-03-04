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

"""LLM-as-judge reward function with optional structured output."""

from __future__ import annotations

import logging
from abc import abstractmethod

from pydantic import BaseModel
from strands import Agent
from strands.models import Model
from typing_extensions import override

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)


class LLMJudgeReward(RewardFunction):
    r"""Abstract base for LLM-as-judge reward functions.

    Subclasses set `judgment_format` class attribute and implement
    `get_judge_prompt` and `get_reward`.

    When `judgment_format` is set, uses structured output and passes
    the parsed Pydantic model to `get_reward`. When `None`, passes
    the raw text response instead.

    Args:
        judge_model: The model to use for judging.
        system_prompt: Optional system prompt for the judge.
        default_reward: Reward to return if the judge fails.

    Example (structured output)::

        class SimpleQAReward(LLMJudgeReward):
            judgment_format = SimpleQAJudgment

            async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
                return f"Question: {action.message}\\nAnswer: {step_result.observation.final_response}"

            async def get_reward(self, judgment: BaseModel | str) -> float:
                return {"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0}[judgment.grade]

    Example (text output)::

        class RegexReward(LLMJudgeReward):
            # judgment_format defaults to None — uses raw text

            async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
                return f"Rate this response 1-10: {step_result.observation.final_response}"

            async def get_reward(self, judgment: BaseModel | str) -> float:
                match = re.search(r"(\\d+)", judgment)
                return int(match.group(1)) / 10 if match else 0.0
    """

    #: Pydantic model for structured output. Subclasses override to enable structured output.
    judgment_format: type[BaseModel] | None = None

    def __init__(
        self,
        judge_model: Model,
        *,
        system_prompt: str | None = None,
        default_reward: float = 0.0,
    ) -> None:
        """Initialize the LLM judge reward function."""
        self.judge_model = judge_model
        self.system_prompt = system_prompt
        self.default_reward = default_reward

    @abstractmethod
    async def get_judge_prompt(self, action: Action, step_result: StepResult) -> str:
        """Format the prompt for the judge model."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def get_reward(self, judgment: BaseModel | str) -> float:
        """Get reward from judgment (structured or text)."""
        raise NotImplementedError("Subclasses must implement this method.")

    @override
    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        try:
            prompt = await self.get_judge_prompt(action, step_result)
        except Exception as e:
            logger.error("Judge prompt rendering failed: %s", e)
            return RewardResult(reward=self.default_reward, info={"reason": "prompt_error", "error": str(e)})

        agent = Agent(model=self.judge_model, system_prompt=self.system_prompt, tools=[])

        try:
            if self.judgment_format is not None:
                judgment: BaseModel | str = await agent.structured_output_async(
                    output_model=self.judgment_format, prompt=prompt
                )
            else:
                result = await agent.invoke_async(prompt)
                judgment = result.message.get("content", [{}])[0].get("text", "")
        except Exception as e:
            logger.error("Judge model invocation failed: %s", e)
            return RewardResult(reward=self.default_reward, info={"reason": "judge_error", "error": str(e)})

        try:
            reward = await self.get_reward(judgment)
        except Exception as e:
            logger.error("Reward computation for judgment failed: %s", e)
            return RewardResult(reward=self.default_reward, info={"reason": "reward_error", "error": str(e)})

        if isinstance(judgment, BaseModel):
            return RewardResult(reward=reward, info={"judgment": judgment.model_dump()})
        return RewardResult(reward=reward, info={"judgment": judgment})
