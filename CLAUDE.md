# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strands-env is an RL environment abstraction for Strands agents — step, observe, reward. It provides a base `Environment` class that wraps a Strands `Agent` with token-level observation tracking (TITO), reward computation, and termination handling. Supports SGLang, Bedrock, and OpenAI model backends.

## Commands

### Setup
```bash
pip install -e ".[dev]"
```

### Linting
```bash
ruff check src/ tests/ examples/
ruff format --check src/ tests/ examples/
mypy src/strands_env
```

### Testing
```bash
# Unit tests (no server needed)
pytest tests/unit/ -v

# Single test
pytest tests/unit/test_environment.py::TestStep::test_successful_step -v

# Unit tests with coverage
pytest tests/unit/ -v --cov=src/strands_env --cov-report=html

# Integration tests (requires running SGLang server; model ID auto-detected via /get_model_info)
# Tests skip automatically if server is unreachable (/health check)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
# Or via env var: SGLANG_BASE_URL=http://localhost:30000 pytest tests/integration/
```

### Integration Tests with Remote GPU Server

```bash
# 1. Launch SGLang on the remote server in docker
ssh <remote-host> "sudo docker run -d --gpus all --name sglang-test -p 30000:30000 --ipc=host lmsysorg/sglang:<tag> python3 -m sglang.launch_server --model-path <model-id> --host 0.0.0.0 --port 30000 --tp <num_gpus> --mem-fraction-static 0.7"
# 2. Tunnel the port locally
ssh -L 30000:localhost:30000 -N -f <remote-host>
# 3. Run tests locally
pytest tests/integration/ -v
```

## Architecture

The package lives in `src/strands_env/` with these modules:

### `core/`

**types.py** — All data types. `Action` carries a user message + `TaskContext` (ground truth, conversation history, arbitrary metadata via `extra="allow"`). `Observation` holds messages, metrics, and optional `TokenObservation` for TITO training. `TerminationReason` maps agent exceptions to enum values via `from_error()` which walks exception cause chains. `StepResult` bundles observation + reward + termination reason.

**models.py** — `ModelFactory = Callable[[], Model]` type and three factory functions (`sglang_model_factory`, `bedrock_model_factory`, `openai_model_factory`). Each returns a zero-arg lambda that creates a fresh Model instance per `step()` call for concurrent isolation. Bedrock and OpenAI remap `max_new_tokens` → `max_tokens` with a shallow dict copy to avoid mutating defaults.

**environment.py** — Base `Environment` class. `step(action)` creates a fresh model via factory, attaches a `TokenManager`, builds an `Agent` with tools/hooks (always includes `ToolLimiter`), runs `invoke_async`, then collects metrics and optional reward. Subclasses override `get_tools()` and `get_hooks()` to customize. Messages are sliced so only new messages from the current step appear in the observation.

### `cli/`

**__init__.py** — CLI entry point with `strands-env` command group. Registers subcommand groups.

**eval.py** — Evaluation CLI commands: `strands-env eval list` shows registered/unavailable benchmarks, `strands-env eval run` executes benchmark evaluation with environment and optional evaluator hooks.

**config.py** — Configuration dataclasses: `SamplingConfig`, `ModelConfig`, `EnvConfig`, `EvalConfig`. Each has `to_dict()` for serialization. Config saved to output directory for reproducibility.

**utils.py** — `build_model_factory(config, max_concurrency)` creates SGLang or Bedrock model factories. `load_env_hook(path)` loads environment hooks. `load_evaluator_hook(path)` loads evaluator hooks. SGLang health check with clear error messages.

### `eval/`

**evaluator.py** — `Evaluator` class orchestrates concurrent rollouts with checkpointing and pass@k metrics. Takes an async `env_factory` for flexible environment creation. Uses tqdm with `logging_redirect_tqdm` for clean progress output. Subclasses implement `load_dataset()` for different benchmarks.

**registry.py** — Benchmark registry with `@register_eval(name)` decorator. Auto-discovers benchmark modules from `benchmarks/` subdirectory on first access. `get_benchmark(name)`, `list_benchmarks()`, and `list_unavailable_benchmarks()` for discovery. Modules with missing dependencies are tracked as unavailable.

**metrics.py** — `compute_pass_at_k` implements the unbiased pass@k estimator. `MetricFn` type alias for pluggable metrics.

**benchmarks/** — Benchmark evaluator modules. Each module uses `@register_eval` decorator. Auto-discovered on first registry access; missing dependencies cause module to be skipped with warning.

**benchmarks/aime.py** — `AIMEEvaluator` base class for AIME benchmarks. `AIME2024Evaluator` and `AIME2025Evaluator` registered as separate benchmarks with different dataset paths.

### `utils/`

**sglang.py** — Sync SGLang server utilities. `check_server_health(base_url)` for early validation. `get_model_id(base_url)` to query the served model. Client/tokenizer caching has moved to `strands_sglang.utils`.

**aws.py** — AWS boto3 session caching. `get_session(region, profile_name, role_arn)` with `@cache`. If `role_arn` provided, uses `RefreshableCredentials` for programmatic role assumption with auto-refresh; otherwise returns basic session.

### `tools/`

**code_interpreter.py** — `CodeInterpreterToolkit` wraps AWS Bedrock AgentCore Code Interpreter. Provides `execute_code` (Python) and `execute_command` (shell) tools. Sessions are lazily created and can be cleaned up via `cleanup()`.

### `environments/`

**calculator/** — `CalculatorEnv` provides a simple calculator tool for math problems. Useful for testing and as a reference implementation.

**code_sandbox/** — `CodeSandboxEnv` uses AWS Bedrock AgentCore Code Interpreter for sandboxed code execution. `CodeMode` enum controls tool availability:
- `CODE`: Only `execute_code` (Python execution)
- `TERMINAL`: Only `execute_command` (shell commands)
- `CODE_AND_TERMINAL`: Both tools

### Key Design Decisions

- **Factory pattern**: `ModelFactory` returns lambdas (not Model instances) so each `step()` gets a fresh model with clean token tracking state.
- **TITO token tracking**: `TokenManager` on SGLang models captures exact token IDs and logprobs during generation. `TokenObservation.from_token_manager()` extracts prompt/rollout split. Non-SGLang models get an empty `TokenManager` (returns `None` from `from_token_manager`).
- **`list()` copies**: Tools, hooks, and messages are copied via `list()` before passing to Agent to prevent cross-step mutation.
- **ToolLimiter**: Always prepended to hooks list. Supports `max_tool_iters` and `max_tool_calls`. Raises `MaxToolIterationsReachedError` or `MaxToolCallsReachedError` which `TerminationReason.from_error()` maps to `MAX_TOOL_ITERATIONS_REACHED` or `MAX_TOOL_CALLS_REACHED`.

## Code Style

- Ruff for linting and formatting (line-length 120, rules: B, D, E, F, G, I, LOG, N, UP, W)
- Pydocstyle with Google convention (enforced in `src/` only)
- Mypy with near-strict settings (see `pyproject.toml` for full config)
- Use lazy `%` formatting for logging (not f-strings)
- Use single backticks `` `xx` `` in docstrings (not Sphinx-style double backticks)
- `__init__` docstrings should be `"""Initialize a `ClassName` instance."""`
- Conventional commits (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- Python 3.10+ required
- asyncio_mode = "auto" for pytest-asyncio
- Async-first: all Environment methods that interact with Agent are async

## Releases

- Do NOT push tags (`git push --tags`) - the user will create GitHub Releases manually to trigger PyPI CI/CD
- When preparing a release: update version in `pyproject.toml`, commit, push code only
- User creates the release on GitHub web UI which triggers the publish workflow
