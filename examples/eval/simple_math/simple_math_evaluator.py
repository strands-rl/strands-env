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

"""Example evaluator hook for a simple math benchmark.

This demonstrates how to create a custom evaluator for use with:
    strands-env eval --evaluator examples/evaluators/simple_math_evaluator.py \
        --env examples/envs/calculator_env.py --backend sglang

The hook file must export `EvaluatorClass` (an Evaluator subclass).
"""

from collections.abc import Iterable

from strands_env.core import Action, TaskContext
from strands_env.eval import Evaluator


class SimpleMathEvaluator(Evaluator):
    """Example evaluator for a simple math benchmark."""

    benchmark_name = "simple-math"

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset and yield Actions for evaluation.

        Replace this with your actual dataset loading logic.
        Each Action should have:
        - message: The problem prompt
        - task_context.id: Unique problem ID
        - task_context.ground_truth: Expected answer
        """
        # Example problems - replace with your dataset
        problems = [
            {
                "id": "prob_1",
                "prompt": "Compute 2 + 3 and 29 * 12, respectively, and add the results.",
                "answer": "353",
            },
            {"id": "prob_2", "prompt": "What is 7 * 8?", "answer": "56"},
            {"id": "prob_3", "prompt": "What is 100 / 4?", "answer": "25"},
        ]

        for item in problems:
            yield Action(
                message=item["prompt"],
                task_context=TaskContext(
                    id=item["id"],
                    ground_truth=item["answer"],
                ),
            )

    def get_metric_fns(self):
        """Return metric functions. Override to customize.

        Default computes pass@k for k=1..n_samples_per_prompt.
        """
        return super().get_metric_fns() + [self.compute_average_reward]

    def compute_average_reward(self, results: dict) -> dict:
        """Example custom metric: average reward across all samples."""
        total_reward = 0.0
        count = 0
        for samples in results.values():
            for sample in samples:
                if sample.step_result.reward:
                    total_reward += sample.step_result.reward.reward
                    count += 1
        avg = total_reward / count if count > 0 else 0.0
        return {"avg_reward": avg}


# Required export - the CLI looks for this
EvaluatorClass = SimpleMathEvaluator
