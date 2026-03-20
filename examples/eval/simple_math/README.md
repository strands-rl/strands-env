# Simple Math

Simple math benchmark using `CalculatorEnv` with a basic calculator tool. Useful as a reference for building custom evaluators.

## Setup

No additional setup required beyond `strands-env`.

## Files

- `calculator_env.py` - Environment hook using `CalculatorEnv`
- `simple_math_evaluator.py` - Custom evaluator hook with example problems

## Usage

```bash
strands-env eval run \
    --evaluator examples.eval.simple_math.simple_math_evaluator \
    --env examples.eval.simple_math.calculator_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
