# SWE-bench

[SWE-bench Verified](https://www.swebench.com/) benchmark using `SWEBenchEnv`, a thin subclass of `TerminalBenchEnv` with a SWE-bench-tuned system prompt. Each task runs in an isolated container with the agent fixing the repository at `/testbed` via `execute_command` tool calls.

## Variants

| Name | Description |
|---|---|
| `swebench-verified` | SWE-bench Verified (Harbor format, sparse-checked out from [harbor-datasets](https://github.com/laude-institute/harbor-datasets)) |

## Setup

1. **Docker** - Must be installed and running
2. **Dependencies** - Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/terminal_bench/requirements.txt
   ```

## Files

- `swe_bench_env.py` - Environment hook that creates `SWEBenchEnv` instances

## Usage

### Docker (default)

```bash
strands-env eval run swebench-verified \
    --env examples.eval.swe_bench.swe_bench_env \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

### EKS/Fargate

Run tasks on AWS EKS/Fargate via [harbor-aws](https://github.com/JackXu0/harbor-aws) by passing `backend` and `eks_backend_config` through `--env-config`:

```bash
export HARBOR_CONTROL_URL=...
export HARBOR_ADMIN_TOKEN=...
strands-env eval run swebench-verified \
    --env examples.eval.swe_bench.swe_bench_env \
    --env-config '{"backend": "eks", "eks_backend_config": {"stack_name": "harbor-aws", "region": "us-east-1", "role_arn": "arn:aws:iam::123456789012:role/harbor-role", "ecr_cache": true}}' \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 89
```

See `strands-env eval run --help` for all CLI options.
