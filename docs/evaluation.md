# Evaluation Guide

This guide covers running benchmark evaluations with `strands-env`.

## CLI Reference

The `strands-env` CLI provides commands for running benchmark evaluations.

### List Benchmarks

```bash
strands-env eval list
```

### Run Evaluation

```bash
# Using a registered benchmark
strands-env eval run <benchmark> --env <dotted.module.path> [options]

# Using a custom evaluator module
strands-env eval run --evaluator <dotted.module.path> --env <dotted.module.path> [options]
```

**Required arguments:**
- `<benchmark>` - Benchmark name (e.g., `aime-2024`, `aime-2025`), OR
- `--evaluator` - Dotted path to evaluator module (exporting `EvaluatorClass`)
- `--env`, `-e` - Dotted path to environment hook module (exporting `create_env_factory`)

**Environment config:**
- `--env-config` - JSON config as inline string or path to JSON file (passed as `**kwargs` to `create_env_factory`)

**Model options:**
- `--backend`, `-b` - Model backend: `sglang` (default), `bedrock`, or `kimi`
- `--base-url` - SGLang server URL (default: `http://localhost:30000`)
- `--model-id` - Model ID (auto-detected for SGLang)
- `--tokenizer-path` - Tokenizer path (defaults to model_id)
- `--tool-parser` - Tool parser name (e.g., `hermes`, `qwen_xml`)
- `--region` - AWS region for Bedrock
- `--profile-name` - AWS profile name for Bedrock
- `--role-arn` - AWS role ARN to assume for Bedrock

**Sampling options:**
- `--temperature` - Sampling temperature
- `--max-tokens` - Maximum new tokens (default: 16384)
- `--top-p` - Top-p sampling
- `--top-k` - Top-k sampling

**Evaluation options:**
- `--n-samples-per-prompt` - Samples per prompt for pass@k (default: 1)
- `--max-concurrency` - Maximum concurrent evaluations (default: 10)
- `--output`, `-o` - Output directory (default: `{benchmark}_eval/`)
- `--max-samples` - Maximum dataset samples to evaluate
- `--save-interval` - Save results every N samples (default: 10)
- `--keep-tokens` - Keep token-level observations in results
- `--debug` - Enable debug logging

### Examples

```bash
# Using registered benchmark with code sandbox env
strands-env eval run aime-2024 \
    --env examples.eval.aime.code_sandbox_env \
    --base-url http://localhost:30000

# Using custom evaluator module
strands-env eval run \
    --evaluator examples.eval.simple_math.simple_math_evaluator \
    --env examples.eval.simple_math.calculator_env \
    --base-url http://localhost:30000

# Pass@8 evaluation with high concurrency
strands-env eval run aime-2024 \
    --env examples.eval.simple_math.calculator_env \
    --base-url http://localhost:30000 \
    --n-samples-per-prompt 8 \
    --max-concurrency 30

# With env config override
strands-env eval run aime-2024 \
    --env examples.eval.simple_math.calculator_env \
    --env-config '{"max_tool_iters": 5}'
```

## Hook Files

Environment hook files define how environments are created for each evaluation sample. They must export a `create_env_factory` function.

### Structure

```python
from strands_env.core.models import ModelFactory

def create_env_factory(model_factory: ModelFactory, **env_config):
    """Create an async environment factory."""
    async def env_factory(action):
        return YourEnvironment(model_factory=model_factory, **env_config)

    return env_factory
```

### Example: Calculator Environment

```python
# examples/eval/simple_math/calculator_env.py
from strands_env.core.models import ModelFactory
from strands_env.environments.calculator import CalculatorEnv
from strands_env.rewards import MathVerifyReward

def create_env_factory(model_factory: ModelFactory, **env_config):
    reward_fn = MathVerifyReward()

    async def env_factory(_action):
        return CalculatorEnv(model_factory=model_factory, reward_fn=reward_fn, **env_config)

    return env_factory
```

### Example: Code Sandbox Environment

```python
# examples/eval/aime_code/code_sandbox_env.py
from strands_env.core.models import ModelFactory
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.rewards import MathVerifyReward

def create_env_factory(model_factory: ModelFactory, **env_config):
    reward_fn = MathVerifyReward()

    async def env_factory(_action):
        return CodeSandboxEnv(model_factory=model_factory, reward_fn=reward_fn, mode="code", **env_config)

    return env_factory
```

## Custom Evaluators

For custom benchmarks, subclass `Evaluator`. You can either register it with `@register_eval` or use an evaluator hook file.

### Evaluator Hook File

Create a Python file that exports `EvaluatorClass`:

```python
# my_evaluator.py
from collections.abc import Iterable

from strands_env.core import Action, TaskContext
from strands_env.eval import Evaluator

class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self) -> Iterable[Action]:
        for item in load_my_data():
            yield Action(
                message=item["prompt"],
                task_context=TaskContext(
                    id=item["id"],
                    ground_truth=item["answer"],
                ),
            )

EvaluatorClass = MyEvaluator
```

Then run:
```bash
strands-env eval run --evaluator my_package.my_evaluator --env my_package.my_env --base-url http://localhost:30000
```

### Registered Evaluator

To add a built-in benchmark, create a module in `src/strands_env/eval/benchmarks/` and use `@register_eval`:

```python
# src/strands_env/eval/benchmarks/my_benchmark.py
from collections.abc import Iterable

from strands_env.core import Action, TaskContext

from ..evaluator import Evaluator
from ..registry import register_eval

@register_eval("my-benchmark")
class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self) -> Iterable[Action]:
        """Load dataset and return Actions for evaluation."""
        for item in load_my_data():
            yield Action(
                message=item["prompt"],
                task_context=TaskContext(
                    id=item["id"],
                    ground_truth=item["answer"],
                ),
            )
```

Benchmarks are auto-discovered from the `benchmarks/` subdirectory. If a benchmark has missing dependencies, it will be listed as unavailable in `strands-env eval list` with the import error message.

### Programmatic Usage

```python
async def run_evaluation():
    evaluator = MyEvaluator(
        env_factory=env_factory,
        n_samples_per_prompt=8,
        max_concurrency=30,
        output_path="results.jsonl",
        keep_tokens=False,
    )

    actions = evaluator.load_dataset()
    results = await evaluator.run(actions)
    metrics = evaluator.compute_metrics(results)
    # {"pass@1": 0.75, "pass@8": 0.95, ...}
```

### Custom Metrics

Override `get_metric_fns()` to customize metrics:

```python
from functools import partial

from strands_env.eval import Evaluator
from strands_env.eval.metrics import compute_pass_at_k

class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self):
        ...

    def get_metric_fns(self):
        # Include default pass@k plus custom metrics
        return [
            partial(compute_pass_at_k, k_values=[1, 5, 10], reward_threshold=1.0),
            self.my_custom_metric,
        ]

    def my_custom_metric(self, results: dict) -> dict:
        return {"my_metric": compute_something(results)}
```

## Tool Parser

For models that use non-standard tool calling formats, specify a predefined parser name from `strands-sglang` via `--tool-parser`:

```bash
strands-env eval run aime-2024 \
    --env examples.eval.simple_math.calculator_env \
    --tool-parser qwen_xml
```

Available parsers: `hermes` (default), `qwen_xml`, `glm`.

## Output Files

Evaluation results are saved to the output directory:

```
{benchmark}_eval/
├── config.json      # CLI configuration for reproducibility
├── results.jsonl    # Per-sample results (action, step_result, reward)
└── metrics.json     # Aggregated metrics (pass@k, etc.)
```

The evaluator supports checkpointing and resume - if interrupted, it will skip already-completed samples on restart.
