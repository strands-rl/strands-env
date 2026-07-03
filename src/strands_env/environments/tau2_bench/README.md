# tau2-bench Environment

`Tau2BenchEnv` runs a [tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research) customer-service task: a multi-turn dialogue between the **agent under test** and an LLM **user-simulator**, both acting on a shared in-memory domain DB. The entire episode runs inside a single `agent.invoke_async()`, driven turn-by-turn by `Tau2BenchUserSimulator` via `AfterInvocationEvent.resume`. Termination is decided by stop markers in the user's reply (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`) or the `max_steps` budget — matching tau2's dual mode, the agent cannot end the dialogue.

## Domains

| Domain | Tasks | Notes |
|---|---|---|
| `airline` | 50 | Agent tools only |
| `retail` | 114 | Agent tools only |
| `telecom` | 114 (sub-sampled from 2285) | User-simulator also gets tools (dual-control devices) |

## Setup

**Install dependencies** — tau2-bench is not on PyPI, so it is installed straight from GitHub, pinned to commit `6899b47` (see `requirements.txt` for why):

```bash
pip install -r requirements.txt
```

The DB, policy, and task data live in the tau2 repo (not the pip wheel); the eval benchmarks clone it into `./data/tau2-bench` at load time.

## Usage

```python
from strands_env.core import Task
from strands_env.environments.tau2_bench import Tau2BenchEnv

env = Tau2BenchEnv(
    agent_model_factory=agent_model_factory,    # the model under test
    user_model_factory=user_model_factory,      # drives the user-simulator
    judge_model_factory=judge_model_factory,    # optional; only used for NL-assertion reward
    domain="retail",
    tau2_task=task_dict,                         # one tau2 `Task`, as a dict
    max_steps=100,
)

await env.reset()                           # Build the per-episode tau2 world
result = await env.rollout(Task(message=""))  # Runs the full multi-turn episode
```

`reset()` builds a fresh tau2 domain environment (applying the task's `initial_state`) and constructs the agent/user tools and the user-simulator. `rollout()` then generates the opening exchange itself — the user-sim replies to a canned greeting, and that reply **replaces** `task.message` (which is why the placeholder is empty). No `cleanup()` is needed — the DB is per-episode in-memory state.

## Tools

The agent's tools are the tau2 domain's own tools, adapted to Strands `AgentTool` via `Tau2BenchTool` and dispatched through tau2's `Environment.get_response` (so `sync_tools` runs after each call). In the `telecom` domain the user-simulator gets its own tool subset (`task.user_tools`). Numeric args to string-typed params are coerced to `str` to match function-calling API behavior (relevant for typeless tool-call formats like GLM).

## Reward

Built-in `Tau2BenchReward` computes the **product** of the sub-rewards named by each task's `reward_basis` (default `{DB, COMMUNICATE}` when unset):

| Basis | Sub-reward |
|---|---|
| `DB` | Agent+user DB hashes match a golden env built by replaying `task.actions` on a fresh DB |
| `ACTION` | Every golden action is matched by some tool call across agent + user-sim messages |
| `COMMUNICATE` | Each required info string appears in some assistant message |
| `NL_ASSERTION` | LLM judge (tau2's judge prompt; documented deviation: snake_case keys, verdict-last field order) grades each expected outcome; needs `judge_model_factory` |
| `ENV_ASSERTION` | Every `task.env_assertions` holds against the live post-episode env (telecom) |

The reward is benchmark material and is not injectable; episodes not ended by the user (`max_steps`, aborts) score 0, matching tau2's dual mode.

## Configuration

Serializable config via `Tau2BenchConfig` (passed as `**kwargs`):

- `domain` — `"airline"`, `"retail"`, or `"telecom"`
- `tau2_task` — one tau2 `Task` serialized to a dict; parsed in `reset()` into `env.tau2_task`
- `max_steps` — step budget in tau2's sense, shared by agent and user-sim (default 100)

Non-serializable params (named args):

- `agent_model_factory` — model factory for the agent under test
- `user_model_factory` — model factory for the user-simulator
- `judge_model_factory` — optional model factory for the NL-assertion judge
