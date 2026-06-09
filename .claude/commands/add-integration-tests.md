Write integration tests for a given environment or feature.

The user provides a target as $ARGUMENTS (e.g., an environment name or class). If not provided, ask.

## Conventions

Follow the existing test style in `tests/integration/`:

- **License header**: Every `.py` file must start with the Apache 2.0 license header (copy from any existing source file)
- **File naming**: `test_<env_name>.py` (e.g., `test_math_env.py`)
- **Location**: `tests/integration/`
- **Docstring**: Start with `"""Integration tests for <ClassName> with a real SGLang model."""`
- **Async tests**: Use `async def test_*` directly — `asyncio_mode = "auto"` is configured
- **Shared fixtures**: `conftest.py` provides `model_factory`, `sglang_base_url`, `sglang_client`, `tokenizer` — use these, don't redefine them
- **Environment setup**: Define a minimal environment class or fixture in the test file, then instantiate it with `model_factory` from conftest
- **Assertions**: Use plain `assert`
- **Extra dependencies**: If the environment has a `requirements.txt` with additional deps, use `pytest.importorskip()` at the top of the test file so tests skip gracefully when deps are missing. Example:
  ```python
  harbor = pytest.importorskip("harbor", reason="harbor required for terminal_bench integration tests")
  ```
  Also note the dependency in the file docstring so users know what to install.

## What to Test

Integration tests validate the **full pipeline** with a real LLM. Each test should cover one observable behavior:

- `test_step_completes` — Agent completes without error, returns `TASK_COMPLETE`
- `test_step_produces_rollout` — SGLang captures token IDs and logprobs
- `test_step_metrics` — Metrics dict has expected keys (`message_count`, `tool_iters`, `model_calls`, etc.)
- `test_reward_computation` — Reward function returns expected values for known inputs
- `test_tool_iteration_limit` — Respects `max_tool_iters` setting
- `test_conversation_history` — Multi-turn works with history passed in `TaskContext`

Do NOT duplicate unit test coverage. Integration tests are expensive (real LLM calls) — keep them focused.

## Steps

1. Read the environment source to understand its tools, reward, and lifecycle
2. Read `tests/integration/conftest.py` for available fixtures
3. Read `tests/integration/test_math_env.py` as the reference example
4. Write tests that exercise the full `reset() -> step() -> cleanup()` cycle
5. After writing, remind the user to run `/run-integration-tests` (requires a running SGLang server)
