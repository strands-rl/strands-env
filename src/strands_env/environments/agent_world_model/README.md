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
from strands_env.environments.agent_world_model import AgentWorldModelEnvironment

env = AgentWorldModelEnvironment(
    model_factory=model_factory,
    scenario="your_scenario",
    envs_path="/path/to/gen_envs.jsonl",
    work_db_path="/path/to/work.db",
    initial_db_path="/path/to/initial.db",
    temp_dir="/path/to/temp_dir",
    max_tool_iters=10,
)
await env.reset()       # starts server + opens MCP session
result = await env.step(action)
await env.cleanup()     # closes session + kills server + removes temp dir
```

`AgentWorldModelRewardFunction` is used by default — no need to pass `reward_fn` unless you want a custom one.

## AgentWorldModelConfig

`AgentWorldModelConfig` extends `EnvironmentConfig` with AWM-specific fields. All config
fields are serializable (strings, ints) and passed as `**kwargs` to the constructor.

| Field | Type | Description |
|---|---|---|
| `scenario` | `str` | Scenario name |
| `envs_path` | `str` | Path to gen_envs.jsonl (contains `scenario`, `db_path`, `full_code`) |
| `work_db_path` | `str` | Working DB copy the server writes to |
| `initial_db_path` | `str` | Read-only DB snapshot (for reward verification) |
| `temp_dir` | `str` | Temp directory for server artifacts (removed on cleanup) |
| `tool_call_timeout` | `int` | MCP tool call timeout in seconds (default: 60) |

## TaskContext Fields

The evaluator/trainer must prepare these fields on `TaskContext` before creating the environment:

| Field | Type | Set by | Used by |
|---|---|---|---|
| `scenario` | `str` | evaluator | env, reward |
| `envs_path` | `str` | evaluator | env |
| `work_db_path` | `str` | evaluator | env, reward |
| `initial_db_path` | `str` | evaluator | reward |
| `temp_dir` | `str` | evaluator | env |
| `verify_code` | `str` | evaluator | reward |
| `task_idx` | `int` | evaluator | reward (logging) |

## Reward

`AgentWorldModelRewardFunction` runs the per-task `verify_task_completion(initial_db_path, final_db_path, final_answer)` function via `exec()`. Each scenario has a unique verification function (from `gen_verifier.pure_code.jsonl`) that checks:

- **DB state changes** — compares initial vs final SQLite database (e.g. "was the item added to cart?")
- **Agent's final answer** — extracts the last assistant message via `Observation.get_final_response()` and validates it (e.g. "is the reported total correct?")

Returns 1.0 if `result["result"] == "complete"`, 0.0 otherwise.

## Lifecycle

- **`reset()`** — Picks a free port, generates and starts a FastAPI server subprocess, waits for TCP readiness, opens an MCP session via `streamable_http_client`, discovers tools as `AgentWorldModelMCPTool` instances.
- **`step(action)`** — Runs the Strands agent with MCP tools. The agent interacts with the FastAPI server to complete the task.
- **`cleanup()`** — Clears tools, closes MCP session/transport (`AsyncExitStack`), kills the server process group (SIGTERM, then SIGKILL after 5s timeout), removes the temp dir.
