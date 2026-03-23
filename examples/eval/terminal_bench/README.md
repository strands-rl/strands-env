# Terminal-Bench

[Terminal-Bench](https://github.com/laude-institute/terminal-bench) benchmark using `TerminalBenchEnv` with Docker-based task execution. Each task runs in an isolated Docker container with the agent interacting via `execute_command` tool calls.

## Variants

| Name | Description |
|---|---|
| `terminal-bench-1` | [Terminal-Bench 1.0](https://github.com/laude-institute/terminal-bench) |
| `terminal-bench-2` | [Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2) |

## Setup

1. **Docker** - Must be installed and running
2. **Dependencies** - Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/terminal_bench/requirements.txt
   ```

## Files

- `terminal_bench_env.py` - Environment hook that creates `TerminalBenchEnv` instances

## Usage

### Docker (default)

```bash
strands-env eval run terminal-bench-2 \
    --env examples.eval.terminal_bench.terminal_bench_env \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

### EKS/Fargate

Run tasks on AWS EKS/Fargate via [harbor-aws](https://github.com/JackXu0/harbor-aws) by passing `backend` and `eks_backend_config` through `--env-config`:

```bash
strands-env eval run terminal-bench-2 \
    --env examples.eval.terminal_bench.terminal_bench_env \
    --env-config '{"backend": "eks", "eks_backend_config": {"stack_name": "harbor-aws", "region": "us-east-1", "role_arn": "arn:aws:iam::123456789012:role/harbor-role"}}' \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 89
```

See `strands-env eval run --help` for all CLI options.
