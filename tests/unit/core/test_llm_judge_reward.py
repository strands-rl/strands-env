from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
from strands.types.exceptions import ModelThrottledException

from strands_env.core.llm_judge_reward import LLMJudgeReward
from strands_env.core.types import RolloutResult, Task

# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class _FakeJudgment(BaseModel):
    grade: str


class _StructuredJudge(LLMJudgeReward[_FakeJudgment]):
    judgment_format = _FakeJudgment

    async def get_judge_prompt(self, task, result):
        return f"Grade this: {result.final_response}"

    async def get_reward(self, judgment):
        return 1.0 if judgment.grade == "correct" else 0.0


class _TextJudge(LLMJudgeReward):
    judgment_format = None

    async def get_judge_prompt(self, task, result):
        return "Grade this"

    async def get_reward(self, judgment):
        return 1.0 if "correct" in judgment else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_and_result():
    task = Task(message="What is 2+2?", ground_truth="4")
    result = RolloutResult(
        messages=[{"role": "assistant", "content": [{"text": "4"}]}],
    )
    return task, result


# ---------------------------------------------------------------------------
# Error recovery paths
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    async def test_prompt_error_returns_default_reward(self):
        """get_judge_prompt raising returns default_reward with judge_prompt_error info."""

        class _FailingPrompt(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, task, result):
                raise ValueError("bad template")

            async def get_reward(self, judgment):
                return 1.0

        judge = _FailingPrompt(judge_model=MagicMock(), default_reward=0.0)
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 0.0
        assert result.info["error_type"] == "judge_prompt_error"

    async def test_judge_agent_error_returns_default_reward(self):
        """A judge that cannot be built scores default_reward with judge_agent_error info.

        `get_system_prompt` is called by `get_judge_agent`, so its failures land here too.
        """

        class _FailingSystemPrompt(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, task, result):
                return "prompt"

            async def get_system_prompt(self, task, result):
                raise RuntimeError("cannot render system prompt")

            async def get_reward(self, judgment):
                return 1.0

        judge = _FailingSystemPrompt(judge_model=MagicMock(), default_reward=0.0)
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 0.0
        assert result.info["error_type"] == "judge_agent_error"

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_get_system_prompt_override_used(self, mock_agent_cls):
        """Overridden get_system_prompt is forwarded to the Agent constructor."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct"}]}
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        class _DynamicPromptJudge(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, task, result):
                return "grade this"

            async def get_system_prompt(self, task, result):
                return f"You are judging task {task.id}"

            async def get_reward(self, judgment):
                return 1.0

        judge = _DynamicPromptJudge(judge_model=MagicMock())
        task, result = _task_and_result()
        await judge.compute(task, result)

        # Verify the dynamic system prompt was passed to Agent
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["system_prompt"].startswith("You are judging task")

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_judge_error_returns_default_reward(self, mock_agent_cls):
        """Agent invocation raising returns default_reward with judge_error info."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke_async = AsyncMock(side_effect=RuntimeError("model down"))
        mock_agent_cls.return_value = mock_agent_instance

        judge = _TextJudge(judge_model=MagicMock(), default_reward=0.5)
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 0.5
        assert result.info["error_type"] == "judge_error"

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_reward_error_returns_default_reward(self, mock_agent_cls):
        """get_reward raising returns default_reward with reward_error info."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "some judgment"}]}
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        class _FailingReward(LLMJudgeReward):
            judgment_format = None

            async def get_judge_prompt(self, task, result):
                return "prompt"

            async def get_reward(self, judgment):
                raise KeyError("unexpected grade")

        judge = _FailingReward(judge_model=MagicMock(), default_reward=0.0)
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 0.0
        assert result.info["error_type"] == "reward_error"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_structured_output_success(self, mock_agent_cls):
        """Structured output mode: judgment_format set, structured output parsed."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.structured_output = _FakeJudgment(grade="correct")
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        judge = _StructuredJudge(judge_model=MagicMock())
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 1.0
        assert result.info["status"] == "success"
        assert result.info["judgment"]["grade"] == "correct"

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_text_output_success(self, mock_agent_cls):
        """Text output mode: judgment_format=None, raw text passed to get_reward."""
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct answer"}]}
        mock_agent_instance.invoke_async = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        judge = _TextJudge(judge_model=MagicMock())
        task, result = _task_and_result()
        result = await judge.compute(task, result)

        assert result.reward == 1.0
        assert result.info["status"] == "success"
        assert result.info["judgment"] == "correct answer"

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_model_list_rotates_per_sample(self, mock_agent_cls):
        """A `judge_model` list round-robins across samples, spreading load over profiles."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct"}]}
        mock_agent_cls.return_value = MagicMock(invoke_async=AsyncMock(return_value=mock_result))

        model_a, model_b = MagicMock(), MagicMock()
        judge = _TextJudge(judge_model=[model_a, model_b])
        for _ in range(3):
            await judge.compute(*_task_and_result())

        used = [call[1]["model"] for call in mock_agent_cls.call_args_list]
        assert used == [model_a, model_b, model_a]

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_throttle_returns_default_reward(self, mock_agent_cls):
        """A throttle surfacing from Strands' own retry is reported as judge_error."""
        throttled = MagicMock(invoke_async=AsyncMock(side_effect=ModelThrottledException("throttled")))
        mock_agent_cls.return_value = throttled

        judge = _TextJudge(judge_model=MagicMock(), default_reward=0.0)
        result = await judge.compute(*_task_and_result())

        assert result.reward == 0.0
        assert result.info["error_type"] == "judge_error"

    @patch("strands_env.core.llm_judge_reward.Agent")
    async def test_get_judge_agent_override_used(self, mock_agent_cls):
        """An overridden get_judge_agent replaces the default agent construction."""
        mock_result = MagicMock()
        mock_result.message = {"content": [{"text": "correct"}]}
        custom = MagicMock(invoke_async=AsyncMock(return_value=mock_result))

        seen = {}

        class _CustomAgentJudge(_TextJudge):
            async def get_judge_agent(self, task, result):
                seen["task"], seen["result"] = task, result
                return custom

        judge = _CustomAgentJudge(judge_model=MagicMock())
        task, rollout = _task_and_result()
        result = await judge.compute(task, rollout)

        assert result.reward == 1.0
        mock_agent_cls.assert_not_called()
        custom.invoke_async.assert_awaited_once()
        # The sample is handed to the override, so an agentic judge can bind per-sample state
        # (e.g. request context) to its tools.
        assert seen["task"] is task
        assert seen["result"] is rollout
