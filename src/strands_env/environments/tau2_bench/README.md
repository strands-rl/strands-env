# tau2-bench Environment

`Tau2BenchEnv` runs a [tau2-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research) customer-service task: a multi-turn dialogue between the **agent under test** and an LLM **user-simulator**, both acting on a shared in-memory domain DB. The entire episode runs inside a single `agent.invoke_async()`, driven turn-by-turn by `Tau2BenchUserSimHook` via `AfterInvocationEvent.resume`. Termination is decided by stop markers in either side's reply (`###STOP###`, `###TRANSFER###`, `###OUT-OF-SCOPE###`) or a `max_turns` cap.

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
from strands_env.environments.tau2_bench import Tau2BenchEnv

env = Tau2BenchEnv(
    agent_model_factory=agent_model_factory,    # the model under test
    user_model_factory=user_model_factory,      # drives the user-simulator
    judge_model_factory=judge_model_factory,    # optional; only used for NL-assertion reward
    initial_db=initial_db,                       # pristine base DB for the domain
    domain="retail",
    task=task_dict,                              # one tau2 `Task`, as a dict
    user_sim_guidelines=guidelines,
    max_turns=100,
)

await env.reset()                # Build per-episode tau2 env, prime the user-sim
result = await env.step(action)  # Runs the full multi-turn episode
```

`reset()` builds a fresh tau2 domain environment (applying `task.initial_state`), constructs the agent/user tools, and gets the user-sim's reply to a canned greeting to seed the agent's first turn. No `cleanup()` is needed — the DB is per-episode in-memory state.

## Tools

The agent's tools are the tau2 domain's own tools, adapted to Strands `AgentTool` via `Tau2BenchTool` and dispatched through tau2's `Environment.get_response` (so `sync_tools` runs after each call). In the `telecom` domain the user-simulator gets its own tool subset (`task.user_tools`). Numeric args to string-typed params are coerced to `str` to match function-calling API behavior (relevant for typeless tool-call formats like GLM).

## Reward

Built-in `Tau2BenchReward` computes the **product** of the sub-rewards named by each task's `reward_basis` (default `{DB, COMMUNICATE}` when unset):

| Basis | Sub-reward |
|---|---|
| `DB` | Agent+user DB hashes match a golden env built by replaying `task.actions` on a fresh DB |
| `ACTION` | Every golden action is matched by some tool call across agent + user-sim messages |
| `COMMUNICATE` | Each required info string appears in some assistant message |
| `NL_ASSERTION` | LLM judge (byte-aligned with tau2's prompt/schema) grades each expected outcome; needs `judge_model_factory` |
| `ENV_ASSERTION` | Every `task.env_assertions` holds against the live post-episode env (telecom) |

Supply a custom `reward_fn` to override.

## Configuration

Serializable config via `Tau2BenchConfig` (passed as `**kwargs`):

- `domain` — `"airline"`, `"retail"`, or `"telecom"`
- `task` — one tau2 `Task` serialized to a dict
- `user_sim_guidelines` — system-prompt guidelines for the user-simulator
- `max_turns` — turn cap before forced termination (default 100)

Non-serializable params (named args):

- `agent_model_factory` — model factory for the agent under test
- `user_model_factory` — model factory for the user-simulator
- `judge_model_factory` — optional model factory for the NL-assertion judge
- `initial_db` — pristine base DB for the domain (deep-copied per episode)
- `reward_fn` — optional override for the default `Tau2BenchReward`
