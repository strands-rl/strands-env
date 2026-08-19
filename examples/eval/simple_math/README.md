# Simple Math

Simple math benchmark using `MathEnv`. Useful as a reference for building custom evaluators.

## Setup

No additional setup required beyond `strands-env`.

## Environments

- `math_env.py` - Environment hook using `MathEnv`
- `simple_math_evaluator.py` - Custom evaluator hook with example problems

## Usage

```bash
python -m strands_env.eval \
    --evaluator examples.eval.simple_math.simple_math_evaluator \
    --env examples.eval.simple_math.math_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
