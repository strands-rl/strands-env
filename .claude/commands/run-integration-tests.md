Run integration tests in an isolated temporary venv created with uv.

Integration tests require a running SGLang server. Before running any commands, ask the user:

1. "What is the SGLang server base URL?" — offer these options:
   - `http://localhost:30000` (default — assumes local or SSH-tunneled server)
   - Let the user provide a custom URL

2. "Which tool parser?" — offer these options:
   - `hermes` (default — for Hermes/Qwen models)
   - `qwen_xml` (XML format for Qwen models)
   - `glm` (GLM models)

Then proceed:

1. Create a temporary venv at `/tmp/strands-env-test-venv` using `uv venv /tmp/strands-env-test-venv --python 3.12 -q`
2. Install the package with dev dependencies: `uv pip install -e ".[dev]" --python /tmp/strands-env-test-venv/bin/python -q`
3. Auto-install environment-specific optional dependencies for tested environments: scan integration test files (`tests/integration/test_*.py`) for imports from `strands_env.environments.<name>`, then for each matched environment check if a corresponding optional dependency group exists in `pyproject.toml` (e.g., `web-search`, `terminal-bench`). Install all matched groups in a single `uv pip install` call (e.g., `uv pip install -e ".[web-search,terminal-bench]"`).
4. Run integration tests with the confirmed URL and tool parser: `/tmp/strands-env-test-venv/bin/python -m pytest tests/integration/ -v --tb=short --sglang-base-url=<URL> --tool-parser=<PARSER> $ARGUMENTS`

IMPORTANT: Never use the active shell's python/pytest — it may point to a different venv. Always use the temporary venv's python.
