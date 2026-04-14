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

"""
Generate function for retool RL training using `strands-env`'s `CodeSandboxEnv`.

Uses:
- AWS Bedrock-based code execution
- Built-in token tracking (TokenObservation)
- Persistent session management
- Tool iteration/call limits
"""

import logging

from slime.rollout.sglang_rollout import GenerateState  # type: ignore
from slime.utils.types import Sample  # type: ignore
from strands_sglang import get_client_from_slime_args

from strands_env.core.models import sglang_model_factory
from strands_env.core.types import Action, TaskContext
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.rewards.math_verify_reward import MathVerifyReward
from strands_env.tools import CodeInterpreterQuotas
from strands_env.utils.aws import get_client
from strands_env.utils.slime import RolloutLogger

# export for slime's --custom-rollout-log-function-path
log_rollout_metrics = RolloutLogger(n_rollouts_per_step=3, log_per_tool_metrics=False).log_rollouts

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a helpful math-solving assistant with access to the `execute_code` tool.

Guidelines:
- For any numerical or symbolic computation, always use the `execute_code` tool rather than performing calculations mentally.
- Break problems into clear steps, calling the Python tool whenever computation is required.
- After completing your reasoning, present the final result enclosed in \\boxed{}.
""".strip()

MAX_TOOL_ITERS = 5
MAX_TOOL_CALLS = None

# Shared quotas enforce account-wide Code Interpreter limits across all concurrent sessions.
QUOTAS = CodeInterpreterQuotas()


async def generate_and_rm(args, sample: Sample, sampling_params) -> Sample:
    """Generate and compute rewards using CodeSandboxEnv."""
    assert not args.partial_rollout, "Partial rollout not supported."

    state = GenerateState(args)

    model_factory = sglang_model_factory(
        tokenizer=state.tokenizer,
        client=get_client_from_slime_args(args, timeout=300.0),
        sampling_params=sampling_params,
    )
    bedrock_client = get_client("bedrock-agentcore")
    reward_fn = MathVerifyReward()
    env = CodeSandboxEnv(
        model_factory=model_factory,
        client=bedrock_client,
        mode="code",
        reward_fn=reward_fn,
        system_prompt=SYSTEM_PROMPT,
        max_tool_iters=MAX_TOOL_ITERS,
        max_tool_calls=MAX_TOOL_CALLS,
        quotas=QUOTAS,
        verbose=False,
    )

    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
    action = Action(
        message=prompt,
        task_context=TaskContext(
            ground_truth=sample.label,
            conversation_history=[],
        ),
    )
    step_result = await env.step(action)

    # Extract token data from observation
    token_obs = step_result.observation.tokens
    sample.tokens = token_obs.token_ids
    sample.loss_mask = token_obs.rollout_loss_mask
    sample.rollout_log_probs = token_obs.rollout_logprobs
    sample.response_length = len(token_obs.rollout_token_ids)
    sample.response = state.tokenizer.decode(token_obs.rollout_token_ids, skip_special_tokens=False)

    # Set status
    if step_result.termination_reason.value == "task_complete":
        sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    # Set step result for custom rollout logging in `log_rollout_metrics`
    sample.step_result = step_result

    await env.cleanup()

    # Compute retool reward
    if step_result.reward.reward == 0.0:
        sample.reward = min(-0.6, -1 + (sample.metrics.get("tool_iters", 0) - 2) / 2 * 0.1)
    else:
        sample.reward = step_result.reward.reward

    return sample
