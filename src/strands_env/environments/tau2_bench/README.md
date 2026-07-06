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
from strands_env.environments.tau2_bench import Tau2BenchEnv, Tau2BenchTask

env = Tau2BenchEnv(
    agent_model_factory=agent_model_factory,    # the model under test
    user_model_factory=user_model_factory,      # drives the user-simulator
    judge_model_factory=judge_model_factory,    # optional; only used for NL-assertion reward
    max_steps=100,
)

task = Tau2BenchTask(domain="retail", config=task_dict)  # one tau2 `Task`, as a dict
result = await env.rollout(task)  # reset (build the tau2 world) -> episode -> reward -> cleanup
```

`rollout()` is a template method: `reset(task)` builds a fresh tau2 domain environment from the task's domain and `initial_state` and constructs the agent/user tools and the user-simulator; the episode then generates the opening exchange itself — the user-sim replies to a canned greeting, and that reply **replaces** `task.message` (which is why `Tau2BenchTask` defaults it to empty).

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

| Field | Default | Meaning |
|---|---|---|
| `max_steps` | `100` | Step budget in tau2's sense, shared by agent and user-sim |

Base knobs come from `EnvironmentConfig`. Non-serializable named args: `agent_model_factory` (the model under test), `user_model_factory` (drives the user-simulator), `judge_model_factory` (optional; NL-assertion judge only).

## Task Fields

| Field | Meaning |
|---|---|
| `domain` | `"airline"`, `"retail"`, or `"telecom"` |
| `config` | One tau2 `Task` serialized to a dict — lazily parsed (`task.tau2_task`) and built into the live world (`task.tau2_env`) as cached properties, so the episode and the reward share the same instance |
