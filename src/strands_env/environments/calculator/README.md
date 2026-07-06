# Calculator Environment

A simple math environment that gives the agent a calculator tool. Useful as a reference implementation and for testing.

## Setup

No additional dependencies required beyond `strands-env`.

## Usage

```python
from strands_env.environments.calculator import CalculatorEnv

env = CalculatorEnv(model_factory=model_factory)
result = await env.rollout(task)
```

## Tools

- **calculator** — Basic arithmetic operations (from `strands_tools`).

## Configuration

No env-specific keys — base knobs (`system_prompt`, `max_tool_iters`, ...) come from `EnvironmentConfig`.

## Reward

`MathVerifyReward` is the default: parses the model's `\boxed{}` answer and checks symbolic equivalence against `task.ground_truth` via HuggingFace `math-verify`. Supply `reward_fn` to override.

## System Prompt

The agent is instructed to solve math problems step by step using the calculator tool, with the final answer in `\boxed{}`.
