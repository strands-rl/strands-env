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
from typing import Generic, cast, override

from pydantic import BaseModel
from strands import Agent
from strands.models import Model
from typing_extensions import TypeVar

from .types import RewardFunction, RewardResult, RolloutResult, TaskT, extract_message_text

logger = logging.getLogger(__name__)

#: TypeVar for the judgment model type. Defaults to `BaseModel` when unparameterized.
JudgmentFormat = TypeVar("JudgmentFormat", bound=BaseModel)


class LLMJudgeReward(RewardFunction[TaskT], Generic[JudgmentFormat, TaskT]):
    r"""Abstract base for LLM-as-judge reward functions.

    Args:
        judge_model: A single model or a list of models to round-robin across
            (useful for spreading load across AWS profiles to avoid throttling).
            The cycle advances per sample, so consecutive judgments hit different
            profiles.
        default_reward: Reward to return if the judge fails.

    Notes:
        - Subclasses set `judgment_format` class attribute and implement
          `get_judge_prompt` and `get_reward`.
        - When `judgment_format` is set, uses structured output and passes
          the parsed Pydantic model to `get_reward`. When `None`, passes
          the raw text response instead.
        - Throttling is handled by Strands' `ModelRetryStrategy`, a default hook on
          every `Agent`: 6 attempts with exponential backoff (`4+8+16+32+64` = ~124s)
          before a `ModelThrottledException` surfaces here as `judge_error`. Override
          `get_judge_agent` to tune it (`retry_strategy=ModelRetryStrategy(...)`), or
          to add a hook that rotates `agent.model` between attempts.

    Example:
        class SimpleQAReward(LLMJudgeReward[SimpleQAJudgment]):
            judgment_format = SimpleQAJudgment

            async def get_judge_prompt(self, task: TaskT, result: RolloutResult) -> str:
                return f"Question: {task.message}\\nAnswer: {result.final_response}"

            async def get_reward(self, judgment: SimpleQAJudgment | str) -> float:
                return {"correct": 1.0, "incorrect": 0.0, "not_attempted": 0.0}[judgment.grade]
    """

    #: Pydantic model for structured output. Subclasses override to enable structured output.
    judgment_format: type[JudgmentFormat] | None = None

    def __init__(
        self,
        judge_model: Model | list[Model],
        *,
        default_reward: float = 0.0,
    ) -> None:
        self.judge_models = itertools.cycle(judge_model if isinstance(judge_model, list) else [judge_model])
        self.default_reward = default_reward

    async def get_system_prompt(self, task: TaskT, result: RolloutResult) -> str | None:
        """Return the system prompt for the judge. Override to set one, or for per-sample prompts."""
        return None

    @abstractmethod
    async def get_judge_prompt(self, task: TaskT, result: RolloutResult) -> str:
        """Format the prompt for the judge model."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    async def get_reward(self, judgment: JudgmentFormat | str) -> float:
        """Get reward from judgment (structured or text)."""
        raise NotImplementedError("Subclasses must implement this method.")

    async def get_judge_agent(self, system_prompt: str | None, name: str = "LLMJudge") -> Agent:
        """Build the agent that judges one sample. Override to configure it."""
        return Agent(model=next(self.judge_models), system_prompt=system_prompt, tools=[], name=name)

    @override
    async def compute(self, task: TaskT, result: RolloutResult) -> RewardResult:
        def _render_error(e: Exception) -> str:
            return f"{type(e).__name__}: {e}"

        # Render system prompt
        try:
            system_prompt = await self.get_system_prompt(task, result)
        except Exception as e:
            logger.error("System prompt rendering failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={"status": "error", "error_type": "system_prompt_error", "error": _render_error(e)},
            )

        # Render judge prompt
        try:
            prompt = await self.get_judge_prompt(task, result)
        except Exception as e:
            logger.error("Judge prompt rendering failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={"status": "error", "error_type": "judge_prompt_error", "error": _render_error(e)},
            )

        # Invoke judge model (Strands' `ModelRetryStrategy` handles throttling internally)
        try:
            agent = await self.get_judge_agent(system_prompt)
            judge_result = await agent.invoke_async(prompt, structured_output_model=self.judgment_format)
            judgment: JudgmentFormat | str = (
                cast(JudgmentFormat, judge_result.structured_output)
                if self.judgment_format is not None
                else extract_message_text(judge_result.message)
            )
        except Exception as e:
            logger.error("Judge model invocation failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={"status": "error", "error_type": "judge_error", "error": _render_error(e)},
            )

        # Get reward
        judgment_data = judgment.model_dump(mode="json") if isinstance(judgment, BaseModel) else judgment
        try:
            reward = await self.get_reward(judgment)
        except Exception as e:
            logger.error("Reward computation for judgment failed: %s", e)
            return RewardResult(
                reward=self.default_reward,
                info={
                    "status": "error",
                    "error_type": "reward_error",
                    "error": _render_error(e),
                    "judgment": judgment_data,
                },
            )

        return RewardResult(reward=reward, info={"status": "success", "judgment": judgment_data})
