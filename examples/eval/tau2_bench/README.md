# tau2-bench

[tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research) multi-turn customer-service benchmark on `Tau2BenchEnv`. Each task runs a full dialogue between the **agent under test** and an LLM **user-simulator** over a shared in-memory domain DB; reward is the product of per-task sub-rewards selected by `reward_basis` (DB-state hash, action match, communicated info, NL-assertion judge, env-assertion).

## Variants

| Name | Description |
|---|---|
| `tau2-bench-retail` | Retail domain (114 tasks) |
| `tau2-bench-airline` | Airline domain (50 tasks) |
| `tau2-bench-telecom` | Telecom domain (114 tasks); user-simulator also has tools |

## Setup

Install tau2 (data files are cloned automatically on first run):

```bash
pip install -r src/strands_env/environments/tau2_bench/requirements.txt
```

## Environments

- `tau2_bench_env.py` — environment hook (`create_env_factory`) that builds `Tau2BenchEnv`, supplying the user-simulator and optional judge model factories and caching the base DB per domain.

## Usage

```bash
python -m strands_env.eval \
    --benchmark tau2-bench-retail \
    --env examples.eval.tau2_bench.tau2_bench_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 20000 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
