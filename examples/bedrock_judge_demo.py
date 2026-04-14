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

"""LLM-as-judge reward demo using Bedrock.

This example demonstrates:
- Creating a custom LLMJudgeRewardFunction subclass
- Using structured output for judgment
- Evaluating responses with an LLM judge

Usage:
    python examples/bedrock_judge_demo.py
    python examples/bedrock_judge_demo.py --model-id us.anthropic.claude-sonnet-4-20250514-v1:0
"""

from __future__ import annotations

import asyncio
import logging

import boto3
import click
from strands.models.bedrock import BedrockModel

from strands_env.core.types import Action, Observation, StepResult, TaskContext, TerminationReason
from strands_env.eval.benchmarks.simpleqa_verified import SimpleQAReward

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

# Simulated responses to evaluate (question, ground_truth, model_response)
TEST_CASES = [
    ("What is the capital of France?", "Paris", "The capital of France is Paris."),
    ("What is 2 + 2?", "4", "2 + 2 equals 5."),
    ("Who wrote Hamlet?", "Shakespeare", "I'm not sure about that question."),
]


def create_mock_step_result(response: str) -> StepResult:
    """Create a mock StepResult with the given response."""
    return StepResult(
        observation=Observation(
            messages=[{"role": "assistant", "content": [{"text": response}]}],
        ),
        termination_reason=TerminationReason.TASK_COMPLETE,
    )


async def run_demo(model_id: str) -> None:
    """Run the LLM judge demo."""
    # Create Bedrock model for judging
    judge_model = BedrockModel(
        model_id=model_id,
        boto_session=boto3.Session(),
        max_tokens=1024,
    )

    # Create reward function
    reward_fn = SimpleQAReward(judge_model=judge_model)

    # Evaluate each test case
    for question, ground_truth, response in TEST_CASES:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Question:     {question}")
        click.echo(f"Ground Truth: {ground_truth}")
        click.echo(f"Response:     {response}")
        click.echo("-" * 60)

        action = Action(message=question, task_context=TaskContext(ground_truth=ground_truth))
        step_result = create_mock_step_result(response)

        result = await reward_fn.compute(action, step_result)

        click.echo(f"Reward:       {result.reward}")
        if result.info.get("status") == "success":
            judgment = result.info["judgment"]
            click.echo(f"Grade:        {judgment.get('grade', 'N/A')}")
        else:
            click.echo(f"Error:        {result.info.get('error_type')}: {result.info.get('error')}")


@click.command()
@click.option(
    "--model-id",
    default="us.anthropic.claude-sonnet-4-20250514-v1:0",
    help="Bedrock model ID for the judge.",
)
def main(model_id: str) -> None:
    """Demo LLM-as-judge reward function with Bedrock."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_demo(model_id))


if __name__ == "__main__":
    main()
