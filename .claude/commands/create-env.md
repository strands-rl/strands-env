Create a new environment skeleton under `src/strands_env/environments/`.

The user provides the environment name as $ARGUMENTS (e.g., `my_env`). If not provided, ask.

Create the following files under `src/strands_env/environments/<name>/`:

1. **`__init__.py`** — License header + docstring + re-export of the environment class from `env.py`.

2. **`env.py`** — License header + minimal `Environment` subclass:
   - Import `Environment` from `strands_env.core.environment`
   - Set `default_system_prompt_path = Path(__file__).parent / "system_prompt.md"`
   - Override `get_tools()` returning an empty list with a TODO comment
   - Override `async def reset(self)` with a TODO comment — resource-heavy or async initialization (e.g., spinning up containers, creating sessions, connecting to services) belongs here, not in `__init__`. `__init__` should only store config and lightweight state. `reset()` is async and called per-episode, making it the right place for setup that needs `await` or fresh-per-episode state.
   - Class name should be `<Name>Env` (e.g., `my_env` → `MyEnvEnv`, `agentcore_code` → `AgentCoreCodeEnv`)

3. **`system_prompt.md`** — A placeholder with a TODO comment for the user to fill in.

4. **`requirements.txt`** — A comment `# No additional dependencies` (user adds deps as needed).

5. **`README.md`** — Template with these sections:
   - `# <Name> Environment` — one-line placeholder description
   - `## Setup` — placeholder
   - `## Usage` — placeholder code snippet with the env class
   - `## Tools` — placeholder
   - `## Reward` — placeholder
   - `## System Prompt` — placeholder

Use the existing environments (calculator, agentcore_code, harbor) as style references. Match the license header used in existing files.

After creating the files, remind the user to:
- Implement `get_tools()` in `env.py`
- Put resource-heavy or async initialization in `reset()`, not `__init__` (e.g., starting containers, creating API sessions). `__init__` is sync-only and should just store config.
- Write the system prompt in `system_prompt.md`
- Add the env to `src/strands_env/environments/__init__.py` if it should be a public export
