# Terminal-Bench

[Terminal-Bench](https://github.com/laude-institute/terminal-bench) benchmark using `HarborEnv` with Docker-based task execution. Each task runs in an isolated Docker container with the agent interacting via `execute_command` tool calls.

## Variants

| Name | Description |
|---|---|
| `terminal-bench-1` | [Terminal-Bench 1.0](https://github.com/laude-institute/terminal-bench) |
| `terminal-bench-2` | [Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2) |
| `terminal-bench-2.1` | [Terminal-Bench 2.1](https://github.com/laude-institute/terminal-bench-2.1) — 89 tasks |

## Setup

1. **Docker** - Must be installed and running
2. **Dependencies** - Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/harbor/requirements.txt
   ```

## Environments

- `terminal_bench_env.py` - Environment hook that creates `HarborEnv` instances

## Usage

### Docker (default)

```bash
python -m strands_env.eval \
    --benchmark terminal-bench-2 \
    --env examples.eval.terminal_bench.terminal_bench_env \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
