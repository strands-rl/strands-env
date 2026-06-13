# SWE-bench

[SWE-bench Verified](https://www.swebench.com/) benchmark using `HarborEnv`. The `swebench-verified` evaluator injects a SWE-bench-tuned system prompt into each task's config. Each task runs in an isolated container with the agent fixing the repository at `/testbed` via `execute_command` tool calls.

## Variants

| Name | Description |
|---|---|
| `swebench-verified` | SWE-bench Verified (Harbor format, sparse-checked out from [harbor-datasets](https://github.com/laude-institute/harbor-datasets)) |

## Setup

1. **Docker** - Must be installed and running
2. **Dependencies** - Install additional requirements:
   ```bash
   pip install -r src/strands_env/environments/harbor/requirements.txt
   ```

## Files

- `swe_bench_env.py` - Environment hook that creates `HarborEnv` instances

## Usage

### Docker (default)

```bash
python -m strands_env.eval \
    --benchmark swebench-verified \
    --env examples.eval.swe_bench.swe_bench_env \
    --base-url http://localhost:30000 \
    --backend sglang \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
