# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strands-env is a framework for building **agent environments** with Strands Agents. An *agent environment* turns a Strands `Agent` into an RL environment whose unit of interaction is a full agent loop (prompt → tool calls → response), not a single model call — with token-level observation tracking (TITO), reward computation, and termination handling. Supports SGLang, Bedrock, OpenAI, and Kimi (Moonshot AI via LiteLLM) model backends.

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
pytest tests/unit/core/test_environment.py::TestRollout::test_successful_rollout -v

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

Holds the environment base, data types, model factories, and shared reward/tool primitives.

**types.py** — All data types. A `Task` is the input to a rollout: an `id`, a user `message`, and a `TaskContext` (ground truth, conversation history, arbitrary metadata via `extra="allow"`). `RolloutResult` is the flattened output of one `rollout()`: `messages`, an optional `rollout` (token-level trajectory) for TITO training, `metrics`, an optional `reward_result` (`RewardResult`), `termination_reason`, and a `final_response` property. `RewardFunction` is the abstract base (async `compute(task, result) -> RewardResult`); `RewardResult` carries `reward` + `info`. `TerminationReason` maps agent exceptions to enum values via `from_error()` which walks exception cause chains.

**models.py** — `ModelFactory = Callable[[], Model]` type and four factory functions (`sglang_model_factory`, `bedrock_model_factory`, `openai_model_factory`, `kimi_model_factory`). Each returns a zero-arg lambda that creates a fresh Model instance per `rollout()` call for concurrent isolation. Bedrock, OpenAI, and Kimi remap `max_new_tokens` → `max_tokens` with a shallow dict copy to avoid mutating defaults. The Kimi factory targets Moonshot AI via LiteLLM (requires `MOONSHOT_API_KEY`) and uses a custom subclass that preserves `reasoning_content` for multi-turn conversations. Also home to the `ModelConfig` dataclass and `build_model_factory(config)` (a `match/case` dispatch over `sglang | bedrock | kimi`), used by the eval CLI. (`openai_model_factory` exists but is not wired into `build_model_factory`.)

**environment.py** — Base `Environment` class. `EnvironmentConfig` TypedDict defines the serializable config shape (`system_prompt`, `max_tool_iters`, `max_tool_calls`, `max_parallel_tool_calls`, `max_messages`, `trace_attributes`, `agent_name`, `verbose`). `__init__` takes `model_factory`, `reward_fn`, and `**config: Unpack[EnvironmentConfig]`. Subclass configs inherit from `EnvironmentConfig` to add env-specific keys. `rollout(task)` creates a fresh model via factory, builds an `Agent` with tools/hooks (always includes `LoopLimiter`), runs `invoke_async`, then reads the model's `Rollout`, collects metrics, and computes the optional reward into a `RolloutResult`. Subclasses override `get_tools()` and `get_hooks()` to customize. Messages are sliced so only new messages from the current rollout appear in the result. Also exports `AsyncEnvFactory` (the async env-factory type used by the evaluator and Ray actors).

**llm_judge_reward.py** — `LLMJudgeReward` abstract base for LLM-as-judge rewards, generic over `JudgmentFormat` (a `TypeVar` bound to `BaseModel`). Subclasses parameterize via `LLMJudgeReward[MyJudgment]` to get typed `get_reward()` signatures. Set class attribute `judgment_format` to a Pydantic model for structured output or leave `None` for raw text. Subclasses implement `get_judge_prompt()` and `get_reward()`. Includes error handling with `default_reward` fallback. (Used by `mcp_atlas`'s per-claim coverage reward.)

**mcp_tool.py** — `MCPToolAdapter`, an `AgentTool` subclass that adapts an MCP tool definition to the Strands agent interface. Handles tool-spec building; subclasses implement `call_tool()` for the transport-specific call and result parsing. Shared base for the `mcp_atlas` (`MCPAtlasTool`, HTTP) and `agent_world_model` (`AgentWorldModelTool`, `ClientSession`) tools.

**distributed.py** — Generic Ray actor pool for distributing `Environment.rollout()` across processes (formerly `utils/ray.py`). `EnvironmentActor` takes `(env_hook_path, env_hook_config)` — loads a callable via dotted path and calls it with the config dict to produce an `AsyncEnvFactory`. `EnvironmentActorPool` distributes actors across Ray nodes with `NodeAffinitySchedulingStrategy`. The actor interface is fully generic; domain-specific adapters (CLI eval, SLiME training) provide the appropriate hook path and config.

### `eval/`

**cli.py** — Evaluation CLI (entry point `python -m strands_env.eval`, wired in `__main__.py`). A single `click` command: `--list` shows registered/unavailable benchmarks; `--benchmark <name>` (or `--evaluator <path>`) runs an evaluation. Env hooks are dotted paths (`--env examples.eval.simple_math.calculator_env`); env config is passed as `--env-config` (inline JSON or path to a JSON file) via a custom `JsonType`. Model backend selected with `--backend` (`sglang` | `bedrock` | `kimi`) plus sampling flags (`--temperature`, `--max-tokens`, `--top-p`, `--top-k`, `--tool-parser`); `build_model_factory` turns these into a `ModelFactory`. Distributed eval via `--n-actors-per-node` creates an `EnvironmentActorPool` (from `core/distributed.py`) backed by Ray.

**__main__.py** — Module entry so `python -m strands_env.eval` invokes the CLI; prepends the cwd to `sys.path` so user-provided hook modules resolve.

**evaluator.py** — `EvalSample` bundles a `RolloutResult` with an `aborted` flag for checkpoint resume. `Evaluator` class orchestrates concurrent rollouts with JSONL checkpointing and pass@k metrics. Takes an `env_factory` (`AsyncEnvFactory`) for local evaluation or an `env_actor_pool` (`EnvironmentActorPool`) for distributed evaluation via Ray. Uses tqdm with `logging_redirect_tqdm` for clean progress output. Subclasses implement `load_dataset()` for different benchmarks and optionally override `validate_sample()` to mark failed samples as aborted (excluded from metrics, retried on resume).

**registry.py** — Benchmark registry with `@register_eval(name)` decorator. Auto-discovers benchmark modules from `benchmarks/` subdirectory on first access. `get_benchmark(name)`, `list_benchmarks()`, and `list_unavailable_benchmarks()` for discovery. Modules with missing dependencies are tracked as unavailable.

**metrics.py** — `compute_pass_at_k` implements the unbiased pass@k estimator. `MetricFunction` type alias for pluggable metrics.

**benchmarks/** — Benchmark evaluator modules. Each module uses `@register_eval` decorator. Auto-discovered on first registry access; missing dependencies cause module to be skipped with warning. Registered families: math (`aime-2024/2025/2026`, `hmmt-feb-2025/nov-2025/feb-2026`), QA/research (`gpqa-*`, `hle-verified-gold[-text]`, `simpleqa-verified`, `frames`, `sealqa-seal-0/hard`, `browsecomp`), instruction following (`ifeval`), MCP (`mcp-atlas`), and agentic-coding/terminal (`swebench-verified`, `terminal-bench-1/2/2.1`).

### `utils/`

**loader.py** — Generic module/function/hook loading utilities (no CLI dependency). `load_module(name)` imports by dotted path. `load_class(name)` and `load_function(name)` import a class or callable by dotted path. `load_env_factory_hook(hook_path)` and `load_evaluator_hook(hook_path)` are convenience wrappers that append the expected attribute name (`.create_env_factory`, `.EvaluatorClass`) and delegate to the generic loaders. Used by both CLI and Ray actors.

**aws.py** — AWS boto3 session and client utilities. `resolve_region_name(...)` resolves the region from explicit arg → `AWS_REGION` → `AWS_DEFAULT_REGION` → profile → `us-east-1`. `get_session(region, profile_name, role_arn)` creates a **fresh** session each call (sessions are not thread-safe). `get_client(service_name, ...)` (cached via `@cache_by`) returns a cached, thread-safe boto3 client (each client gets its own dedicated session). If `role_arn` provided, uses `RefreshableCredentials` for programmatic role assumption with auto-refresh.

**decorators.py** — `@requires_env(*env_vars)` validates environment variables at call time (async functions return an error string on missing vars; sync functions raise `OSError`). `@with_timeout(seconds)` enforces a timeout via `ThreadPoolExecutor`. `@cache_by(*key_args)` caches a function's result keyed by selected named arguments.

**slime_logger.py** — `RolloutLogger` for `slime` training, with a pluggable backend. Aggregates per-rollout env metrics into slime's `rollout_extra_metrics` and publishes a sample of decoded rollouts to the configured `backend` (`"wandb"` → metrics to W&B + samples to a Weave dataset; `"mlflow"` → metrics and JSON sample artifacts to MLflow). Backend libraries are imported lazily. Pass its bound `log_rollouts` as slime's `--custom-rollout-log-function-path` callback.

### `environments/`

Each environment is a package: `env.py` holds the `Environment` subclass, plus domain-named helpers as needed (`server.py`, `quotas.py`, etc.). **Layout convention for tools and rewards**: one of each → a singular module (`tool.py`, `reward.py`); several → a plural subpackage (`tools/`, `rewards/`) whose members keep domain names (`tools/search.py`, not `tools/tool1.py`). Pay the rename on the 1→N transition rather than pre-creating an empty package — this matches the repo's flatten-by-default preference.

**calculator/** — `CalculatorEnv` provides a simple calculator tool for math problems; defaults its reward to `MathVerifyReward` (`reward.py`). `MathVerifyReward` gives reward 1.0 if the model's `\boxed{}` answer is mathematically equivalent to ground truth, using the `math_verify` library for SymPy-based symbolic equivalence (fractions, sets, simplification). Parses only the tail of the response to avoid long chain-of-thought. Useful for testing and as a reference implementation.

**agentcore_code/** — `AgentCoreCodeEnv` uses AWS Bedrock AgentCore Code Interpreter for sandboxed code execution. `AgentCoreCodeConfig` extends `EnvironmentConfig` with `mode: Literal["code", "terminal", "code_and_terminal"]`. The tool (`tool.py`) is `CodeInterpreterToolkit`, which provides `execute_code` (Python) and `execute_command` (shell) tools; sessions are lazily created and cleaned up via `cleanup()`. `quotas.py` holds AgentCore service-quota helpers.

**mcp_atlas/** — `MCPAtlasEnv`, the MCP-Atlas benchmark env backed by a Docker container (default `http://localhost:1984`) exposing an MCP-tool REST API. A shared `httpx.AsyncClient` is passed at construction (caller owns its lifecycle); `reset()` fetches tools from the container and applies per-task filtering (caching `/list-tools` across episodes); `cleanup()` clears the tool list. `MCPAtlasConfig` adds `enabled_tools` and `tool_timeout`. The tool (`tool.py` → `MCPAtlasTool`) is an `MCPToolAdapter` that POSTs to `/call-tool`. `reward.py` is a per-claim LLM-as-judge (`LLMJudgeReward` subclass) following MCP-Atlas coverage scoring (fulfilled/partial/not → 1.0/0.5/0.0, averaged; pass threshold 0.75).

**agent_world_model/** — `AgentWorldModelEnv` backed by a per-task FastAPI + SQLite server subprocess (`server.py` generates and launches a self-contained script). The agent talks to the server over MCP via `AgentWorldModelTool` (`tool.py`, an `MCPToolAdapter` over a `ClientSession` that polls the server process to fail fast on exit). `AgentWorldModelConfig` adds `scenario`, `envs_path`, `work_db_path`, `initial_db_path`, `temp_dir`, and optional `tool_call_timeout`. `reward.py` (`AgentWorldModelReward`) runs each task's `verify_task_completion` via `exec()` against the work DB for a binary reward. Depends on the external `awm` package.

**web_search/** — `WebSearchEnv` with pluggable search providers. Its two tools live in `tools/` (`tools/search.py` → `WebSearchToolkit`, `tools/scrape.py` → `WebScraperToolkit`) per the plural-subpackage convention. `WebSearchToolkit` exposes Serper and Google Custom Search providers as separate `@tool` methods over a shared aiohttp session, with `apply_blocked_domains` for domain filtering and lazy credential validation via `@requires_env`. `WebScraperToolkit` fetches pages via the Jina Reader API (`https://r.jina.ai/{url}`, gated by `@requires_env("JINA_API_KEY")`) and optionally extracts a structured `WebPageSummary` (rationale/evidence/summary) via an LLM summarizer using `Agent.structured_output_async`; `token_budget` truncates via tiktoken cl100k encoding; ported from OpenSeeker. `WebSearchConfig` extends `EnvironmentConfig` with search/scrape settings (`search_provider`, `search_timeout`, `blocked_domains`, `scrape_enabled`, `scrape_timeout`, `scrape_token_budget`). Non-serializable params (`search_concurrency`, `scrape_concurrency`, `summarizer_model_factory`) are named args.

**harbor/** — `HarborEnv` runs any Harbor-format task (a directory with `task.toml`, `environment/Dockerfile`, `tests/test.sh`) in a local Docker container or self-hosted e2b sandbox. `HarborConfig` extends `EnvironmentConfig` with `task_id`, `task_dir`, `trial_dir`, `task_env_config`, `timeout`, `backend` (`"docker"` | `"e2b"`), and `prebaked_e2b_config`. Provides a single `execute_command` tool for shell commands in the container. `HarborReward` (`reward.py`) uploads and runs verification tests (`tests/test.sh`) in the sandbox for binary (0/1) reward. `e2b.py` holds `PrebakedE2BEnvironment`, an `E2BEnvironment` subclass that boots from a pre-baked `template_id` (resolved via a `templates.json`/`E2B_TEMPLATES_PATH` mapping) instead of Harbor's auto-build route, for self-hosted e2b clusters. Both the `terminal-bench-*` and `swebench-verified` eval benchmarks run on this single env — they differ only in dataset and system prompt (the SWE-bench evaluator injects its prompt via the serializable `system_prompt` config key, so there is no separate SWE env class).

**tau2_bench/** — `Tau2BenchEnv`, a thin wrapper over the external `tau2` package for tau2-bench tasks (domains `airline`/`retail`/`telecom`). Multi-turn user interaction is driven by `Tau2BenchUserSimulator` (`simulator.py`), an after-invocation hook owning the user-side agent (persona prompt, user tools, dialogue memory) that produces the next user message via `event.resume`. `max_steps` mirrors tau2's step semantics (total message count across both agents), enforced as one budget: a `LoopLimiter` on the user-sim agent is re-armed each turn with what the assistant conversation has not consumed. Matching tau2 dual mode, only the user can end the dialogue (`###STOP###`/`###TRANSFER###`/`###OUT-OF-SCOPE###`). Takes separate `agent_model_factory` and `user_model_factory` (plus optional `judge_model_factory`) as named args; `Tau2BenchConfig` adds `domain`, `tau2_task` (serialized tau2 Task, parsed once in `__init__`), and optional `max_steps`; user-sim guidelines are derived inside the simulator (tools variant iff the user has tools), mirroring tau2's own UserSimulator. `reset()` builds the per-episode tau2 env and exposes its (agent and user) tools as `Tau2BenchTool` (`tool.py`, an `AgentTool` subclass) instances. `reward.py` (`Tau2BenchReward`) computes the final reward as the product of sub-rewards selected by the task's `reward_basis`; its `Tau2BenchNLAssertionReward` sub-reward (an `LLMJudgeReward[NLJudgment]`) is byte-aligned with tau2's NL-assertion judge prompt/schema. `Tau2BenchReward` first gates on a clean stop — only `user_stop` is scored, a premature end such as `max_steps` scores 0 (matching tau2's `evaluate_simulation`). All tau2 imports are funnelled through a single private `_tau2.py` shim (PEP 562 lazy `__getattr__` for classes + helper functions for the dynamic per-domain modules): `tau2` freezes `TAU2_DATA_DIR` at import time, so the package stays import-pure and nothing should `import tau2` directly elsewhere (a unit test enforces this). Depends on the external `tau2` package.

### Key Design Decisions

- **Factory pattern over raw Model**: Always use our `ModelFactory` functions (`sglang_model_factory`, `bedrock_model_factory`, etc.) instead of constructing Strands `Model` classes directly. The factories handle per-backend concerns that raw constructors don't: `max_new_tokens` → `max_tokens` remapping, shared boto3 client reuse across instances, SGLang client/tokenizer wiring, and consistent sampling param handling. `ModelFactory` returns lambdas (not Model instances) so each `rollout()` gets a fresh model with clean token tracking state.
- **TITO token tracking**: SGLang models accumulate a `Rollout` (token IDs, loss mask, logprobs, segment info) during generation, read via `agent.model.rollout` in `rollout()`. Non-SGLang models yield an empty `Rollout` (falsy via `__len__`), so `RolloutResult.rollout` carries token data only on the SGLang backend.
- **`list()` copies**: Tools, hooks, and messages are copied via `list()` before passing to Agent to prevent cross-rollout mutation.
- **LoopLimiter**: Always prepended to hooks list. Supports `max_tool_iters`, `max_tool_calls`, `max_parallel_tool_calls`, and `max_messages`. Raises `MaxToolIterationsReachedError`, `MaxToolCallsReachedError`, or `MaxMessagesReachedError` which `TerminationReason.from_error()` maps to `MAX_TOOL_ITERATIONS_REACHED`, `MAX_TOOL_CALLS_REACHED`, or `MAX_MESSAGES_REACHED`.
- **TypedDict configs**: Environment configs use `TypedDict` with `Unpack` for `**kwargs` typing. Base `EnvironmentConfig` defines common serializable fields; subclass configs (e.g., `AgentCoreCodeConfig`, `WebSearchConfig`, `HarborConfig`, `MCPAtlasConfig`, `AgentWorldModelConfig`) inherit and add env-specific keys. Non-serializable dependencies (`model_factory`, `reward_fn`, semaphores, etc.) stay as named params. The `self.config` dict stores the full config for subclass access and serialization. **Design rule**: if a parameter is JSON-serializable (strings, ints, bools, lists), it goes in the TypedDict; if it's not (callables, semaphores, clients), it's a named `__init__` param. This enables passing env config as JSON via CLI (`--env-config`) or across process boundaries (Ray actors).
- **Dotted path hooks**: Environment and evaluator hooks are loaded by dotted module path (e.g., `examples.eval.simple_math.calculator_env`), not file paths. The `utils/loader.py` module provides generic loading utilities shared by CLI and Ray actors.
- **MCP tool adapter**: MCP-backed environments share `core/mcp_tool.py`'s `MCPToolAdapter` (an `AgentTool` subclass) rather than the Strands `MCPClient`. The base handles tool-spec building; each env subclasses `call_tool()` for its transport (`mcp_atlas` → HTTP to a container; `agent_world_model` → an MCP `ClientSession` to a subprocess).

## Code Style

- Ruff for linting and formatting (line-length 120, rules: B, D, E, F, G, I, LOG, N, UP, W)
- Pydocstyle with Google convention (enforced in `src/` only)
- Mypy with near-strict settings (see `pyproject.toml` for full config)
- Use lazy `%` formatting for logging (not f-strings)
- Use single backticks `` `xx` `` in docstrings (not Sphinx-style double backticks)
- `__init__` docstrings should be `"""Initialize a `ClassName` instance."""`
- Conventional commits (feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert)
- Python 3.12+ required (CI tests 3.12 and 3.13)
- asyncio_mode = "auto" for pytest-asyncio
- Async-first: all Environment methods that interact with Agent are async

### Class Attribute Conventions

Three forms, chosen by where the value's storage lives and whether the name binds a single value:

- **Bare class-body annotation** (`tau2_env: Tau2Environment`) — instance attribute born later
  (usually in `reset(task)`), PEP 526's "instance variable without default". Non-Optional for mypy;
  accessing before birth raises AttributeError loudly. Never `ClassVar`.
- **`NAME: ClassVar[T] = value` (UPPER_SNAKE)** — true constant: one value program-wide, readers may
  inline it mentally; `ClassVar` forbids instance shadowing (a fidelity hazard for benchmark
  material like prompts and stop patterns). `DEFAULT_*` uppercase = the single canonical fallback
  value. On pydantic models/dataclasses `ClassVar` is mandatory (else it becomes a field).
- **`name: T = value` (lowercase)** — subclass-overridable knob (`benchmark_name`, `domain`,
  `default_system_prompt_path`): the value is polymorphic across subclasses, so it is not a
  constant to the reader. Declare (with annotation) once where the name is introduced — usually the
  base class; subclasses override with bare assignments, no re-annotation (the declaration is
  inherited and mypy checks assignments against it). Add `ClassVar` when per-instance variation has
  its own channel (e.g. `system_prompt` config vs `default_system_prompt_path`).

Enum members are the exception to everything above: bare UPPER assignments, never annotated.

## Releases

- Do NOT push tags (`git push --tags`) - the user will create GitHub Releases manually to trigger PyPI CI/CD
- When preparing a release: update version in `pyproject.toml`, commit, push code only
- User creates the release on GitHub web UI which triggers the publish workflow

## Maintenance

When adding new modules, changing commands, or altering key design patterns, update this file to reflect those changes.
