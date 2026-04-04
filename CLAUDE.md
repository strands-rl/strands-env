# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strands-env is an RL environment abstraction for Strands agents — step, observe, reward. It provides a base `Environment` class that wraps a Strands `Agent` with token-level observation tracking (TITO), reward computation, and termination handling. Supports SGLang, Bedrock, OpenAI, and Kimi (Moonshot AI via LiteLLM) model backends.

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
pytest tests/unit/core/test_environment.py::TestStep::test_successful_step -v

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

**models.py** — `ModelFactory = Callable[[], Model]` type and four factory functions (`sglang_model_factory`, `bedrock_model_factory`, `openai_model_factory`, `kimi_model_factory`). Each returns a zero-arg lambda that creates a fresh Model instance per `step()` call for concurrent isolation. Bedrock, OpenAI, and Kimi remap `max_new_tokens` → `max_tokens` with a shallow dict copy to avoid mutating defaults. The Kimi factory targets Moonshot AI via LiteLLM (requires `MOONSHOT_API_KEY`) and uses a custom subclass that preserves `reasoning_content` for multi-turn conversations.

**environment.py** — Base `Environment` class. `EnvironmentConfig` TypedDict defines the serializable config shape (`system_prompt`, `max_tool_iters`, `max_tool_calls`, `max_parallel_tool_calls`, `verbose`). `__init__` takes `model_factory`, `reward_fn`, and `**config: Unpack[EnvironmentConfig]`. Subclass configs inherit from `EnvironmentConfig` to add env-specific keys. `step(action)` creates a fresh model via factory, attaches a `TokenManager`, builds an `Agent` with tools/hooks (always includes `ToolLimiter`), runs `invoke_async`, then collects metrics and optional reward. Subclasses override `get_tools()` and `get_hooks()` to customize. Messages are sliced so only new messages from the current step appear in the observation.

### `cli/`

**__init__.py** — CLI entry point with `strands-env` command group. Registers subcommand groups.

**eval.py** — Evaluation CLI commands: `strands-env eval list` shows registered/unavailable benchmarks, `strands-env eval run` executes benchmark evaluation. Env hooks are specified as dotted paths (`--env examples.eval.simple_math.calculator_env`). Environment config is passed as `--env-config` (inline JSON or path to JSON file) via custom `JsonType` click parameter. Supports distributed evaluation via `--n-actors-per-node` which creates an `EnvironmentActorPool` backed by Ray. `create_distributed_env_factory()` bridges CLI's `ModelConfig` to the eval hook contract for use inside Ray actors.

**models.py** — `SamplingParams` and `ModelConfig` dataclasses. `ModelConfig` includes `max_connections` for SGLang client pooling. `build_model_factory(config)` creates SGLang, Bedrock, or Kimi model factories with `match/case` dispatch.

### `eval/`

**evaluator.py** — `EvalSample` bundles step result with an `aborted` flag for checkpoint resume. `Evaluator` class orchestrates concurrent rollouts with JSONL checkpointing and pass@k metrics. Takes an `env_factory` (`AsyncEnvFactory`) for local evaluation or an `env_actor_pool` (`EnvironmentActorPool`) for distributed evaluation via Ray. Uses tqdm with `logging_redirect_tqdm` for clean progress output. Subclasses implement `load_dataset()` for different benchmarks and optionally override `validate_sample()` to mark failed samples as aborted (excluded from metrics, retried on resume).

**registry.py** — Benchmark registry with `@register_eval(name)` decorator. Auto-discovers benchmark modules from `benchmarks/` subdirectory on first access. `get_benchmark(name)`, `list_benchmarks()`, and `list_unavailable_benchmarks()` for discovery. Modules with missing dependencies are tracked as unavailable.

**metrics.py** — `compute_pass_at_k` implements the unbiased pass@k estimator. `MetricFn` type alias for pluggable metrics.

**benchmarks/** — Benchmark evaluator modules. Each module uses `@register_eval` decorator. Auto-discovered on first registry access; missing dependencies cause module to be skipped with warning.

**benchmarks/aime.py** — `AIMEEvaluator` base class for AIME benchmarks. `AIME2024Evaluator`, `AIME2025Evaluator`, and `AIME2026Evaluator` registered as separate benchmarks with different dataset paths.

**benchmarks/simpleqa_verified.py** — `SimpleQAVerifiedEvaluator` for the [google/simpleqa-verified](https://huggingface.co/datasets/google/simpleqa-verified) factuality benchmark. `SimpleQAJudgment` Pydantic model for structured grading (CORRECT/INCORRECT/NOT_ATTEMPTED). `SimpleQAReward(LLMJudgeReward[SimpleQAJudgment])` uses OpenAI's grading prompt template.

**benchmarks/frames.py** — `FramesEvaluator` for the [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) multi-hop reasoning benchmark (824 samples). `FramesJudgment` with TRUE/FALSE grading. `FramesReward(LLMJudgeReward[FramesJudgment])`. Wiki links and reasoning types stored in `TaskContext` extras.

**benchmarks/browsecomp.py** — `BrowseCompEvaluator` for [BrowseComp](https://openai.com/index/browsecomp/) browsing agent benchmark. Dataset loaded from OpenAI's public CSV with XOR decryption of encrypted problems/answers. `BrowseCompJudgment` with yes/no grading. `BrowseCompReward(LLMJudgeReward[BrowseCompJudgment])` uses OpenAI's grader template.

### `utils/`

**loader.py** — Generic module/function/hook loading utilities (no CLI dependency). `load_module(name)` imports by dotted path. `load_class(name)` and `load_function(name)` import a class or callable by dotted path. `load_env_factory_hook(hook_path)` and `load_evaluator_hook(hook_path)` are convenience wrappers that append the expected attribute name (`.create_env_factory`, `.EvaluatorClass`) and delegate to the generic loaders. Used by both CLI and Ray actors.

**ray.py** — Generic Ray actor pool for distributing `Environment.step()` across processes. `EnvironmentActor` takes `(env_hook_path, env_hook_config)` — loads a callable via dotted path and calls it with the config dict to produce an `AsyncEnvFactory`. `EnvironmentActorPool` distributes actors across Ray nodes with `NodeAffinitySchedulingStrategy`. The actor interface is fully generic; domain-specific adapters (CLI eval, SLiME training) provide the appropriate hook path and config.

**sglang.py** — Sync SGLang server utilities. `check_server_health(base_url)` for early validation. `get_model_id(base_url)` to query the served model. Client/tokenizer caching has moved to `strands_sglang.utils`.

**aws.py** — AWS boto3 session and client utilities. `get_session(region, profile_name, role_arn)` creates a **fresh** session each call (sessions are not thread-safe). `get_client(service_name, ...)` with `@cache` returns a cached, thread-safe boto3 client (each client gets its own dedicated session). If `role_arn` provided, uses `RefreshableCredentials` for programmatic role assumption with auto-refresh.

### `tools/`

**code_interpreter.py** — `CodeInterpreterToolkit` wraps AWS Bedrock AgentCore Code Interpreter. Provides `execute_code` (Python) and `execute_command` (shell) tools. Sessions are lazily created and can be cleaned up via `cleanup()`.

**web_search.py** — `WebSearchToolkit` with Serper and Google Custom Search providers, shared aiohttp session, and `concurrency: Semaphore | int` for API rate limiting. Domain blocking via `apply_blocked_domains` static method. Credentials validated lazily via `@requires_env` decorator at call time.

**web_scraper.py** — `WebScraperToolkit` using trafilatura (primary) or html2text (fallback) for content extraction. Optional LLM-based summarization via a `summarizer_model_factory`. `concurrency: Semaphore | int` for rate limiting. Token budget (default 5000) limits extracted content length via tiktoken encoding.

### `rewards/`

**llm_judge_reward.py** — `LLMJudgeReward` abstract base for LLM-as-judge rewards, generic over `JudgmentFormat` (a `TypeVar` bound to `BaseModel`). Subclasses parameterize via `LLMJudgeReward[MyJudgment]` to get typed `get_reward()` signatures. Set class attribute `judgment_format` to a Pydantic model for structured output or leave `None` for raw text. Subclasses implement `get_judge_prompt()` and `get_reward()`. Includes error handling with `default_reward` fallback.

**math_verify_reward.py** — `MathVerifyReward` gives reward 1.0 if the model's `\boxed{}` answer is mathematically equivalent to ground truth. Uses `math_verify` library for SymPy-based symbolic equivalence (fractions, sets, simplification). Parses only the tail of the response to avoid long chain-of-thought.

### `environments/`

**calculator/** — `CalculatorEnv` provides a simple calculator tool for math problems. Useful for testing and as a reference implementation.

**code_sandbox/** — `CodeSandboxEnv` uses AWS Bedrock AgentCore Code Interpreter for sandboxed code execution. `CodeSandboxConfig` extends `EnvironmentConfig` with `mode: Literal["code", "terminal", "code_and_terminal"]`.

**mcp/** — `MCPEnvironment` backed by a single MCP server. Manages `MCPClient` lifecycle via `reset()` (starts client) and `cleanup()` (stops client). Supports optional pre-constructed client or subclass-managed initialization.

**web_search/** — `WebSearchEnv` with pluggable search providers. `WebSearchConfig` extends `EnvironmentConfig` with search/scrape settings (`search_provider`, `search_timeout`, `blocked_domains`, `scrape_enabled`, `scrape_timeout`, `scrape_token_budget`). Non-serializable params (`search_concurrency`, `scrape_concurrency`, `summarizer_model_factory`) are named args.

**terminal_bench/** — `TerminalBenchEnv` using Harbor's `DockerEnvironment` for task execution. `TerminalBenchConfig` extends `EnvironmentConfig` with `task_id`, `task_dir`, `trial_dir`, `harbor_env_config`, `timeout`. Provides `execute_command` tool for shell commands in Docker. `TerminalBenchReward` runs verification tests in the container for binary (0/1) reward.

### Other `utils/`

**decorators.py** — `@requires_env(*env_vars)` decorator that validates environment variables at call time. Works with both sync and async functions: async functions return an error string on missing vars; sync functions raise `OSError`. `@with_timeout(seconds)` enforces a timeout on function execution using `ThreadPoolExecutor`.

### Key Design Decisions

- **Factory pattern over raw Model**: Always use our `ModelFactory` functions (`sglang_model_factory`, `bedrock_model_factory`, etc.) instead of constructing Strands `Model` classes directly. The factories handle per-backend concerns that raw constructors don't: `max_new_tokens` → `max_tokens` remapping, shared boto3 client reuse across instances, SGLang client/tokenizer wiring, and consistent sampling param handling. `ModelFactory` returns lambdas (not Model instances) so each `step()` gets a fresh model with clean token tracking state.
- **TITO token tracking**: `TokenManager` on SGLang models captures exact token IDs and logprobs during generation. `TokenObservation.from_token_manager()` extracts prompt/rollout split. Non-SGLang models get an empty `TokenManager` (returns `None` from `from_token_manager`).
- **`list()` copies**: Tools, hooks, and messages are copied via `list()` before passing to Agent to prevent cross-step mutation.
- **ToolLimiter**: Always prepended to hooks list. Supports `max_tool_iters` and `max_tool_calls`. Raises `MaxToolIterationsReachedError` or `MaxToolCallsReachedError` which `TerminationReason.from_error()` maps to `MAX_TOOL_ITERATIONS_REACHED` or `MAX_TOOL_CALLS_REACHED`.
- **TypedDict configs**: Environment configs use `TypedDict` with `Unpack` for `**kwargs` typing. Base `EnvironmentConfig` defines common serializable fields; subclass configs (e.g., `CodeSandboxConfig`, `WebSearchConfig`, `TerminalBenchConfig`) inherit and add env-specific keys. Non-serializable dependencies (`model_factory`, `reward_fn`, semaphores, etc.) stay as named params. The `self.config` dict stores the full config for subclass access and serialization. **Design rule**: if a parameter is JSON-serializable (strings, ints, bools, lists), it goes in the TypedDict; if it's not (callables, semaphores, clients), it's a named `__init__` param. This enables passing env config as JSON via CLI (`--env-config`) or across process boundaries (Ray actors).
- **Dotted path hooks**: Environment and evaluator hooks are loaded by dotted module path (e.g., `examples.eval.simple_math.calculator_env`), not file paths. The `utils/loader.py` module provides generic loading utilities shared by CLI and Ray actors.

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

## Maintenance

When adding new modules, changing commands, or altering key design patterns, update this file to reflect those changes.
