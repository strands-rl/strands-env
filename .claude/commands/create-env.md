Create a new environment skeleton under `src/strands_env/environments/`.

The user provides the environment name as $ARGUMENTS (e.g., `my_env`). If not provided, ask.

Create the following files under `src/strands_env/environments/<name>/`:

1. **`__init__.py`** — License header + docstring + re-exports (env class, config, and task class if any).

2. **`env.py`** — License header + minimal `Environment` subclass:
   - `class <Name>Config(EnvironmentConfig)` — run-level knobs only (each field gets an inline comment with its default). Per-sample values do NOT go here — they belong on the task type.
   - `class <Name>Env(Environment)` — or `Environment[<Name>Task]` when the env has a typed task; the generic parametrization auto-derives `task_cls`.
   - `__init__` stores run-level config and lightweight state only (sync, no `await`). No docstring unless parameters carry semantics (then a full `Args:` section).
   - `async def reset(self, task)` — episode init from the typed sample: containers, sessions, per-episode scratch (`mkdtemp`, cloned working state). Paired with `cleanup()`, which must tolerate partially-initialized state.
   - Override `get_tools()` returning an empty list with a TODO comment.
   - Class name: `my_env` → `MyEnvEnv`, `agentcore_code` → `AgentCoreCodeEnv`.

3. **`task.py`** (only if the env has per-sample fields — the bar is "real types or real consumers"):
   - Module docstring: `"""The per-sample input type for `<Name>Env`."""`
   - `class <Name>Task(Task)` — class docstring says what one sample IS; declared fields with `Field(description=...)` carrying semantics. Tasks may own data-derived views of their fields (path wrappers), never scoring behavior.

4. **`system_prompt.md`** — A placeholder with a TODO comment.

5. **`requirements.txt`** — `# No additional dependencies` (user adds deps as needed).

6. **`README.md`** — the house skeleton:
   - `# <Name> Environment` — one-paragraph description
   - `## Setup` (only if extra deps/creds)
   - `## Usage` — capability-only construction + typed task + `result = await env.rollout(task)` (never manual `reset()`/`cleanup()`)
   - `## Configuration` — table (Field | Default | Meaning) + one line pointing base knobs at `EnvironmentConfig`
   - `## Task Fields` — table (typed-task envs only)
   - `## Tools`
   - `## Reward`
   - `## System Prompt` or `## Lifecycle` (only where reset/cleanup is non-trivial)

Follow CLAUDE.md's "Docstring Style" and "Class Attribute Conventions" sections. Use harbor and agent_world_model as style references (the most recently normalized). Match the license header used in existing files.

After creating the files, remind the user to:
- Implement `get_tools()` and `reset(task)`/`cleanup()`
- Write the system prompt
- Add the env to `src/strands_env/environments/__init__.py` if it should be a public export
