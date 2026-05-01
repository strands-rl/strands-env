# strands-env

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

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

## Built-in Environments

Ready-to-use environments under `src/strands_env/environments/`. Each ships with its own README, system prompt, and `requirements.txt`.

| Environment | Description |
| --- | --- |
| [`calculator`](src/strands_env/environments/calculator/README.md) | Simple environment with a calculator tool for math reasoning. |
| [`code_sandbox`](src/strands_env/environments/code_sandbox/README.md) | Sandboxed Python / shell execution via AWS Bedrock AgentCore Code Interpreter. |
| [`web_search`](src/strands_env/environments/web_search/README.md) | Pluggable search (Serper / Google CSE) + Jina-based page scraping with optional LLM summarization, enlightened by [OpenSeeker](https://github.com/rui-ye/OpenSeeker). |
| [`terminal_bench`](src/strands_env/environments/terminal_bench/README.md) | Run [Terminal-Bench](https://www.tbench.ai/) tasks against a [Harbor](https://github.com/harbor-framework/harbor)-managed Docker/EKS container. |
| [`swe_bench`](src/strands_env/environments/swe_bench/) | [SWE-bench](https://www.swebench.com/) task runner — thin subclass of `terminal_bench` with a SWE-bench-tuned system prompt. |
| [`mcp_atlas`](src/strands_env/environments/mcp_atlas/README.md) | [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) benchmark runner across 36 MCP servers with 500 tasks. |
| [`agent_world_model`](src/strands_env/environments/agent_world_model/README.md) | [AgentWorldModel](https://github.com/scaleapi/agent-world-model) tasks with 1000 synthetic FastAPI + SQLite environments exposed as MCP tools. |

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
