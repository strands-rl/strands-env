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
from strands_env.environments.agent_world_model import AgentWorldModelEnv

env = AgentWorldModelEnv(
    model_factory=model_factory,
    scenario="your_scenario",
    envs_path="/path/to/gen_envs.jsonl",
    work_db_path="/path/to/work.db",
    initial_db_path="/path/to/initial.db",
    temp_dir="/path/to/temp_dir",
    max_tool_iters=10,
)
result = await env.rollout(task)  # reset (start server + open MCP session) -> episode -> reward -> cleanup
```

`AgentWorldModelReward` is used by default — no need to pass `reward_fn` unless you want a custom one.

## Configuration

`AgentWorldModelConfig` keys (passed as `**kwargs`):

| Field | Default | Meaning |
|---|---|---|
| `scenario` | required | Scenario name |
| `envs_path` | required | Path to `gen_envs.jsonl` (contains `scenario`, `db_path`, `full_code`) |
| `work_db_path` | required | Working DB copy the server writes to |
| `initial_db_path` | required | Read-only DB snapshot (for reward verification) |
| `temp_dir` | required | Temp directory for server artifacts (removed on cleanup) |
| `tool_call_timeout` | `60` | MCP tool call timeout in seconds |

Base knobs (`system_prompt`, `max_tool_iters`, `max_tool_calls`, `max_parallel_tool_calls`, `max_messages`, `trace_attributes`, `agent_name`, `verbose`) come from `EnvironmentConfig`.

## Task Fields

The evaluator/trainer must prepare these fields on the `Task` (as extras) before `rollout()`:

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

`AgentWorldModelReward` runs the per-task `verify_task_completion(initial_db_path, final_db_path, final_answer)` function via `exec()`. Each scenario has a unique verification function (from `gen_verifier.pure_code.jsonl`) that checks:

- **DB state changes** — compares initial vs final SQLite database (e.g. "was the item added to cart?")
- **Agent's final answer** — extracts the last assistant message via `RolloutResult.final_response` and validates it (e.g. "is the reported total correct?")

Returns 1.0 if `result["result"] == "complete"`, 0.0 otherwise.

## Lifecycle

- **`reset()`** — Picks a free port, generates and starts a FastAPI server subprocess, waits for TCP readiness, opens an MCP session via `streamable_http_client`, discovers tools as `AgentWorldModelTool` instances.
- **`rollout(task)`** — Runs the Strands agent with MCP tools. The agent interacts with the FastAPI server to complete the task.
- **`cleanup()`** — Clears tools, closes MCP session/transport (`AsyncExitStack`), kills the server process group (SIGTERM, then SIGKILL after 5s timeout), removes the temp dir.
