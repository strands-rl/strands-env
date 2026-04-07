# strands-env

[![CI](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml/badge.svg)](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-env.svg)](https://pypi.org/project/strands-env/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/horizon-rl/strands-env)

Standardizing environment infrastructure with [Strands Agents](https://github.com/strands-agents/sdk-python) — step, observe, reward.

## Features

This package treats each `env.step()` as a **full agent loop** `(prompt → (tool_call, tool_response)* → response)`, not a single model call.

- **Define Environments** — Subclass `Environment`, add `@tool` functions, plug in `RewardFunction`
- **RL Training** — Token-level observations for on-policy training with [strands-sglang](https://github.com/horizon-rl/strands-sglang)
- **Benchmarking** — CLI and `Evaluator` with checkpointing, resume, and custom metrics

## Install

```bash
pip install strands-env
```

For development:

```bash
git clone https://github.com/horizon-rl/strands-env.git && cd strands-env
pip install -e ".[dev]"
```

## Quick Start

### Define an Environment

Subclass `Environment` and add tools as `@tool`-decorated functions:

```python
from strands import tool
from strands_env.core import Environment

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

class MathEnv(Environment):
    def get_tools(self):
        return [calculator]
```

### Run It

```python
env = MathEnv(model_factory=factory, reward_fn=reward_fn)
result = await env.step(Action(message="What is 2^10?", task_context=TaskContext(ground_truth="1024")))

result.observation.final_response   # "The answer is 1024"
result.reward.reward                # 1.0
result.termination_reason           # TerminationReason.TASK_COMPLETE
```

See [`examples/calculator_demo.py`](examples/calculator_demo.py) for a complete example.

### Run Evaluations

```bash
strands-env eval aime-2024 \
    --env examples.eval.simple_math.calculator_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --n-samples-per-prompt 8 \
    --max-concurrency 30
```

> **Tip:** For a non-agentic benchmark (no tool use), simply don't override `get_tools()` in your environment — the base class returns `[]` by default.

## Documentation

- [Evaluation Guide](docs/evaluation.md) — CLI reference, hook files, custom evaluators
- [RL Training Integration](docs/rl-training.md) — slime integration, token observations

## Development

```bash
# Lint
ruff check src/ && ruff format --check src/

# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running SGLang server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
```

Or if using Claude Code, just use `/run-unit-tests` and `/run-integration-tests` slash commands.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
