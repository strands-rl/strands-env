# RL Training Integration

This guide covers integrating `strands-env` with RL training frameworks, specifically [slime](https://github.com/THUDM/slime/).

## Overview

`strands-env` captures token-level observations (TITO) during agent rollouts, which are essential for on-policy RL training. The `TokenObservation` contains:
- `token_ids` - All tokens (prompt + rollout)
- `rollout_token_ids` - Only generated tokens
- `rollout_logprobs` - Log probabilities for each generated token
- `rollout_loss_mask` - Mask for loss computation

## slime Integration

Customize the `generate` (and optionally `reward_func`) entry points to replace single-shot generation with an agentic rollout:

```python
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from strands_sglang import get_client_from_slime_args

from strands_env.core import Action, TaskContext
from strands_env.core.models import sglang_model_factory

async def generate_and_rm(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)  # provides cached tokenizer + slime state

    model_factory = sglang_model_factory(
        tokenizer=state.tokenizer,
        client=get_client_from_slime_args(args, timeout=300.0),
        sampling_params=sampling_params,
    )

    env = YourEnv(model_factory=model_factory, reward_fn=YourRewardFunction())
    prompt = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[0]["content"]
    action = Action(
        message=prompt,
        task_context=TaskContext(ground_truth=sample.label, conversation_history=[]),
    )
    step_result = await env.step(action)

    # Extract TITO data for training
    token_obs = step_result.observation.tokens
    sample.tokens = token_obs.token_ids
    sample.loss_mask = token_obs.rollout_loss_mask
    sample.rollout_log_probs = token_obs.rollout_logprobs
    sample.response_length = len(token_obs.rollout_token_ids)
    sample.response = state.tokenizer.decode(token_obs.rollout_token_ids, skip_special_tokens=False)

    # Status + reward
    sample.status = (
        Sample.Status.COMPLETED
        if step_result.termination_reason.value == "task_complete"
        else Sample.Status.TRUNCATED
    )
    sample.reward = step_result.reward.reward
    sample.step_result = step_result  # for custom rollout logging

    await env.cleanup()
    return sample
```

If you want to compute the reward asynchronously (separate from the rollout), split the work into `generate` + `reward_func`:

```python
async def reward_func(args, sample, **kwargs):
    reward_fn = YourRewardFunction()
    reward_result = await reward_fn.compute(action=sample.action, step_result=sample.step_result)
    return reward_result.reward
```

A complete worked example lives at `examples/slime/retool/generate_with_code_sandbox.py`.

## Key Points

- **Connection pooling**: `get_client_from_slime_args(args)` provides `lru_cache`-backed connection pooling across rollouts for efficient GPU utilization
- **Token observations**: `TokenObservation` contains token IDs and logprobs for on-policy training (SGLang backend only)
- **Model factory pattern**: Each `step()` creates a fresh model instance for clean token tracking state
- **Cleanup**: Call `await env.cleanup()` at the end of each rollout for envs that hold external resources (e.g. `CodeSandboxEnv`, `MCPEnvironment`, `TerminalBenchEnv`)
- **Custom rollout logging**: Attach `step_result` to the sample and use `strands_env.utils.slime.RolloutLogger` (see the retool example) to log per-step metrics via slime's `--custom-rollout-log-function-path`
