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

## Reward

No built-in reward function. Supply a custom `reward_fn` or use with an evaluator (e.g., exact-match on `\boxed{}` answers).

## System Prompt

The agent is instructed to solve math problems step by step using the calculator tool, with the final answer in `\boxed{}`.
