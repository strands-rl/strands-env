# Math Environment

A chat-only math environment: the model reasons in text and boxes its answer, with no tools in the loop. Useful as a reference implementation, for testing, and as the baseline to compare a tool-using math env against.

## Setup

No additional dependencies required beyond `strands-env`.

## Usage

```python
from strands_env.environments.math import MathEnv

env = MathEnv(model_factory=model_factory)
result = await env.rollout(task)
```

## Tools

None. `get_tools()` returns an empty list.

## Configuration

No env-specific keys — base knobs (`system_prompt`, `max_tool_iters`, ...) come from `EnvironmentConfig`.

## Reward

`MathVerifyReward` is the default: parses the model's `\boxed{}` answer and checks symbolic equivalence against `task.ground_truth` via HuggingFace `math-verify`. Supply `reward_fn` to override.

## System Prompt

The agent is instructed to solve math problems step by step, with the final answer in `\boxed{}`.
