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

"""Unit tests for LLMJudgeReward error recovery and happy paths."""

from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
from strands.types.exceptions import ModelThrottledException

from strands_env.core.types import Action, Observation, StepResult, TaskContext
from strands_env.rewards.llm_judge_reward import LLMJudgeReward

# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class _FakeJudgment(BaseModel):
    grade: str


class _StructuredJudge(LLMJudgeReward[_FakeJudgment]):
    judgment_format = _FakeJudgment

    async def get_judge_prompt(self, action, step_result):
        return f"Grade this: {step_result.observation.final_response}"

    async def get_reward(self, judgment):
        return 1.0 if judgment.grade == "correct" else 0.0


class _TextJudge(LLMJudgeReward):
    judgment_format = None

    async def get_judge_prompt(self, action, step_result):
        return "Grade this"

    async def get_reward(self, judgment):
        return 1.0 if "correct" in judgment else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _action_and_step():
    action = Action(message="What is 2+2?", task_context=TaskContext(ground_truth="4"))
    step_result = StepResult(
        observation=Observation(messages=[{"role": "assistant", "content": [{"text": "4"}]}]),
    )
    return action, step_result


# ---------------------------------------------------------------------------
# Error recovery paths
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    async def test_prompt_error_returns_default_reward(self):
        """get_judge_prompt raising returns default_reward with prompt_error info."""

        class _FailingPrompt(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, action, step_result):
                raise ValueError("bad template")

            async def get_reward(self, judgment):
                return 1.0

        judge = _FailingPrompt(judge_model=MagicMock(), default_reward=0.0)
        action, step_result = _action_and_step()
        result = await judge.compute(action, step_result)

        assert result.reward == 0.0
        assert result.info["error_type"] == "prompt_error"

    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_judge_error_returns_default_reward(self, mock_agent_cls):
        """Agent invocation raising returns default_reward with judge_error info."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke_async = AsyncMock(side_effect=RuntimeError("model down"))
        mock_agent_cls.return_value = mock_agent_instance

        judge = _TextJudge(judge_model=MagicMock(), default_reward=0.5)
        action, step_result = _action_and_step()
        result = await judge.compute(action, step_result)

        assert result.reward == 0.5
        assert result.info["error_type"] == "judge_error"

    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_reward_error_returns_default_reward(self, mock_agent_cls):
        """get_reward raising returns default_reward with reward_error info."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "some judgment"}]}
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        class _FailingReward(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, action, step_result):
                return "prompt"

            async def get_reward(self, judgment):
                raise KeyError("unexpected grade")

        judge = _FailingReward(judge_model=MagicMock(), default_reward=0.0)
        action, step_result = _action_and_step()
        result = await judge.compute(action, step_result)

        assert result.reward == 0.0
        assert result.info["error_type"] == "reward_error"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_structured_output_success(self, mock_agent_cls):
        """Structured output mode: judgment_format set, structured output parsed."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.structured_output_async = AsyncMock(
            return_value=_FakeJudgment(grade="correct"),
        )
        mock_agent_cls.return_value = mock_agent_instance

        judge = _StructuredJudge(judge_model=MagicMock())
        action, step_result = _action_and_step()
        result = await judge.compute(action, step_result)

        assert result.reward == 1.0
        assert result.info["status"] == "success"
        assert result.info["judgment"]["grade"] == "correct"

    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_text_output_success(self, mock_agent_cls):
        """Text output mode: judgment_format=None, raw text passed to get_reward."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct answer"}]}
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        judge = _TextJudge(judge_model=MagicMock())
        action, step_result = _action_and_step()
        result = await judge.compute(action, step_result)

        assert result.reward == 1.0
        assert result.info["status"] == "success"
        assert result.info["judgment"] == "correct answer"

    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_throttle_rotates_to_next_model(self, mock_agent_cls):
        """On throttle, cycles to the next model and succeeds."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct"}]}
        throttled = MagicMock(invoke_async=AsyncMock(side_effect=ModelThrottledException("throttled")))
        ok = MagicMock(invoke_async=AsyncMock(return_value=mock_result))
        mock_agent_cls.side_effect = [throttled, ok]

        model_a, model_b = MagicMock(), MagicMock()
        judge = _TextJudge(judge_model=[model_a, model_b], max_model_retries=2)
        result = await judge.compute(*_action_and_step())

        assert result.reward == 1.0
        assert mock_agent_cls.call_args_list[0][1]["model"] is model_a
        assert mock_agent_cls.call_args_list[1][1]["model"] is model_b

    @patch("strands_env.rewards.llm_judge_reward.Agent")
    async def test_all_retries_throttled_returns_default(self, mock_agent_cls):
        """When all retries exhausted, returns default_reward."""
        throttled = MagicMock(invoke_async=AsyncMock(side_effect=ModelThrottledException("throttled")))
        mock_agent_cls.return_value = throttled

        judge = _TextJudge(judge_model=MagicMock(), max_model_retries=2, default_reward=0.0)
        result = await judge.compute(*_action_and_step())

        assert result.reward == 0.0
        assert result.info["error_type"] == "judge_error"
