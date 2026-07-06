# AgentWorldModel Environment

MCP environment for [AgentWorldModel](https://github.com/scaleapi/agent-world-model) tasks — 1000 synthetic FastAPI + SQLite environments exposed as MCP tools via `fastapi_mcp`.

## Setup

**Install additional dependencies**:
```bash
pip install -r requirements.txt
```
This installs [AgentWorldModel](https://pypi.org/project/agent-world-model/) (`agent_world_model`) for scenario generation and verification.

## Usage

```python
from strands_env.environments.agent_world_model import AgentWorldModelEnv, AgentWorldModelTask

env = AgentWorldModelEnv(
    model_factory=model_factory,
    envs_path="/path/to/gen_envs.jsonl",
    max_tool_iters=10,
)

task = AgentWorldModelTask(
    message="Add two apples to the cart and report the total.",
    scenario="your_scenario",
    task_idx=0,
    verify_code=verify_code,          # from gen_verifier.pure_code.jsonl
    initial_db_path="/path/to/initial.db",
)
result = await env.rollout(task)  # reset (clone DB + start server) -> episode -> reward -> cleanup
```

`AgentWorldModelReward` is built in — it is tied to the env's working DB, so there is no `reward_fn` parameter.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `envs_path` | required | Path to `gen_envs.jsonl` (contains `scenario`, `db_path`, `full_code`) |
| `tool_call_timeout` | `60` | MCP tool call timeout in seconds |

Base knobs (`system_prompt`, `max_tool_iters`, ...) come from `EnvironmentConfig`.

## Task Fields

`AgentWorldModelTask` carries the per-sample payload:

| Field | Meaning |
|---|---|
| `scenario` | Scenario name in `gen_envs.jsonl` (which synthetic world to boot) |
| `task_idx` | Task index within the scenario (used in logs) |
| `verify_code` | Python source defining `verify_task_completion(...)` |
| `initial_db_path` | Pristine SQLite snapshot; the episode's working DB is cloned from it |

## Tools

Dynamic — discovered per episode from the scenario's FastAPI server via MCP (`fastapi_mcp`); each endpoint of the synthetic world becomes one tool.

## Reward

`AgentWorldModelReward` runs the task's `verify_task_completion(initial_db_path, final_db_path, final_answer)` via `exec()`. Each scenario has a unique verification function (from `gen_verifier.pure_code.jsonl`) that checks:

- **DB state changes** — compares the initial snapshot against the episode's working DB (e.g. "was the item added to cart?")
- **Agent's final answer** — validates `RolloutResult.final_response` (e.g. "is the reported total correct?")

Returns 1.0 if `result["result"] == "complete"`, 0.0 otherwise. `info["status"]` is `"success"` whenever the verifier ran (reward 0.0 = the agent failed the task) and `"error"` only for verification-machinery failures — the same contract as the other environments.

## Lifecycle

- **`reset(task)`** — Creates a fresh scratch dir (`mkdtemp`), clones `task.initial_db_path` into it as the working DB (per-episode isolation by construction — concurrent episodes never share mutable state), generates and starts a FastAPI server subprocess on the clone, waits for TCP readiness, opens an MCP session, discovers tools.
- **`rollout(task)`** — Runs the full episode: `reset`, the agent loop against the MCP tools, reward, `cleanup`.
- **`cleanup()`** — Clears tools, closes the MCP session/transport, kills the server process group (SIGTERM, then SIGKILL), removes the scratch dir it created.
