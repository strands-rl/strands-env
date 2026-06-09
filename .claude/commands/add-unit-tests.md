Write unit tests for a given module or class.

The user provides a target as $ARGUMENTS (e.g., a file path, class name, or module). If not provided, ask.

## Conventions

Follow the existing test style in `tests/unit/`:

- **License header**: Every `.py` file must start with the Apache 2.0 license header (copy from any existing source file)
- **File naming**: `test_<module>.py` (e.g., `test_environment.py` for `core/environment.py`)
- **Location**: `tests/unit/`
- **Docstring**: Start with `"""Unit tests for <ClassName>."""` or `"""Tests for <module>."""`
- **Test classes**: Group by feature/method, named `Test<ClassName><Feature>` (e.g., `TestEnvironmentInit`, `TestStep`)
- **Test methods**: `test_<behavior>` in snake_case, no docstrings unless the behavior is non-obvious
- **Async tests**: Use `async def test_*` directly — `asyncio_mode = "auto"` is configured in `pyproject.toml`
- **Fixtures**: Define at the top of the file after imports, separated by `# ---------------------------------------------------------------------------`
- **Mocking**: Use `unittest.mock` (`MagicMock`, `AsyncMock`, `patch`). For model factories, use `MagicMock` that returns a mock model with `rollout = Rollout()`
- **Imports**: Import from the public API (e.g., `from strands_env.core.types import Action`) not internal modules
- **Assertions**: Use plain `assert`, not `self.assertEqual`

## Scope

- Focus on the **public API** — constructor, public methods, properties
- Do NOT test private methods (`_ensure_*`, `_internal_helper`) directly
- Do NOT over-test — if a behavior is already covered by integration tests, skip it
- Keep test-to-source ratio reasonable (~1:1 or less)
- Each test should test ONE behavior

## Steps

1. Read the source file to understand the public interface
2. Read existing tests in `tests/unit/` for style reference if needed
3. Write tests covering: constructor defaults, constructor with custom args, key public methods, edge cases, error cases
4. After writing, run tests with `/run-unit-tests` to verify they pass
