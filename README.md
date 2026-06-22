# strands-env

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

[![CI](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml/badge.svg)](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-env.svg)](https://pypi.org/project/strands-env/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/horizon-rl/strands-env)

A framework for building agent environments for RL training and evaluation with Strands Agents.

## Features

An **agent environment** takes a task and runs the agent to completion over multiple turns, producing a **rollout result** — the trajectory, reward, and termination reason for that task. With `strands-env`, you can:

- **Define Environments** — Subclass `Environment`, add `@tool` functions, plug in `RewardFunction`
- **RL Training** — Token-level trajectories (TITO) for on-policy training with [strands-sglang](https://github.com/horizon-rl/strands-sglang)
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
import subprocess
import sys

from strands import tool
from strands_env.core import Environment

@tool
def run_python(code: str) -> str:
    """Run a Python snippet and return its output."""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    return proc.stdout + proc.stderr

class CodingEnv(Environment):
    def get_tools(self):
        return [run_python]
```

### Run It

```python
from strands_env.core import Task, TaskContext

env = CodingEnv(model_factory=factory, reward_fn=reward_fn)
result = await env.rollout(Task(
    message="Write Python to compute the 10th Fibonacci number, then run it.",
    context=TaskContext(ground_truth="55"),
))

result.final_response       # "The 10th Fibonacci number is 55"
result.reward_result        # {"reward": 1.0, "info": ...}
result.termination_reason   # TerminationReason.TASK_COMPLETE
```

See the [`examples/`](examples/) directory for complete, runnable demos.

### Run Evaluations

```bash
python -m strands_env.eval \
    --benchmark terminal-bench-2 \
    --env examples.eval.terminal_bench.terminal_bench_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --n-samples-per-prompt 4 \
    --max-concurrency 8
```

> Raise `--n-samples-per-prompt` for more stable pass@k, and `--max-concurrency` if you're using a hosted sandbox service.

> **Tip:** For a non-agentic benchmark (no tool use), don't override `get_tools()` — the base class returns `[]` by default.

## Built-in Environments

Ready-to-use environments under `src/strands_env/environments/`. Each ships with its own README, system prompt, and `requirements.txt`.

| Environment | Description |
| --- | --- |
| [`calculator`](src/strands_env/environments/calculator/README.md) | Simple environment with a calculator tool for math reasoning. |
| [`harbor`](src/strands_env/environments/harbor/README.md) | Run [Harbor](https://github.com/laude-institute/harbor)-format tasks in sandboxes. Supports training like [SETA](https://github.com/camel-ai/seta) and evaluation like [Terminal-Bench](https://www.tbench.ai/) and [SWE-bench](https://www.swebench.com/). |
| [`agentcore_code`](src/strands_env/environments/agentcore_code/README.md) | Python / shell execution via AWS Bedrock AgentCore Code Interpreter. |
| [`web_search`](src/strands_env/environments/web_search/README.md) | Google search + Jina page scraping with optional LLM summarization, enlightened by [OpenSeeker](https://github.com/rui-ye/OpenSeeker). |
| [`mcp_atlas`](src/strands_env/environments/mcp_atlas/README.md) | [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) benchmark runner across 36 MCP servers with 500 tasks. |
| [`agent_world_model`](src/strands_env/environments/agent_world_model/README.md) | [AgentWorldModel](https://github.com/scaleapi/agent-world-model) tasks with 1000 synthetic FastAPI + SQLite environments exposed as MCP tools. |

## Documentation

- [Evaluation Guide](docs/evaluation.md) — CLI reference, hook files, custom evaluators
- [RL Training Integration](docs/rl-training.md) — integration with the slime RL training framework

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
