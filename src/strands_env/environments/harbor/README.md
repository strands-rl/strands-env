# Harbor Task Environment

`HarborEnv` runs any [Harbor](https://github.com/laude-institute/harbor)-format task in an isolated container. Each task is a directory with an `environment/Dockerfile`, a `tests/test.sh` verification script, and an instruction; the agent solves it by issuing shell commands via a single `execute_command` tool.

Both [Terminal-Bench](https://github.com/laude-institute/terminal-bench) and [SWE-bench](https://www.swebench.com/) tasks follow this exact contract, so they both run on `HarborEnv` — they differ only in their dataset and system prompt, which the eval benchmarks (`terminal-bench-1`, `terminal-bench-2`, `swebench-verified`) supply.

## Backends

| Backend | Description |
|---|---|
| `"docker"` (default) | Local Docker via Harbor's `DockerEnvironment` |
| `"e2b"` | Self-hosted e2b sandbox (Firecracker microVM, e2b-on-AWS) |

## Setup

1. **Docker** (for `"docker"` backend) — Must be installed and running:
   ```bash
   docker info  # verify Docker is available
   ```

2. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Task data** — Each task requires a Harbor-format directory:
   ```
   task_dir/
   ├── task.toml          # task metadata
   ├── instruction.md     # task instruction
   ├── environment/
   │   └── Dockerfile     # container image
   └── tests/
       └── test.sh        # verification script (writes reward.txt)
   ```

## Usage

```python
from strands_env.environments.harbor import HarborEnv, HarborTask

env = HarborEnv(model_factory=model_factory)  # capability only: backend / exec_timeout / prebaked_e2b_config

task = HarborTask.from_task_dir("/path/to/task", trial_dir="/path/to/output")  # reads task.toml
result = await env.rollout(task)  # reset (build + start container) -> episode -> reward -> cleanup
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `backend` | `"docker"` | `"docker"` or `"e2b"` |
| `exec_timeout` | `1200` | Seconds per `sandbox.exec` command; also the verifier fallback |
| `prebaked_e2b_config` | `{}` | e2b connection + template map (see `PrebakedE2BConfig`) |

Base knobs come from `EnvironmentConfig`.

## Task Fields

| Field | Meaning |
|---|---|
| `task_id` | Harbor task name (also keys e2b template lookups) |
| `task_dir` | Path to the task bundle (`task.toml`, `environment/`, `tests/`) |
| `trial_dir` | Output directory for this trial's artifacts (the evaluator decides the layout) |
| `task_env_config` | Container settings from `task.toml` `[environment]` |
| `verifier_timeout` | Verifier budget from `task.toml` `[verifier]`; `None` = `exec_timeout` |
| `system_prompt` | Optional benchmark prompt override (e.g. the SWE-bench-tuned prompt) |

`HarborTask.from_task_dir()` is the canonical bundle→task mapping; `task.task_paths` / `task.trial_paths` expose harbor's own path views.

## Tools

- **execute_command** — Execute any shell command inside the container.

## Reward

Built-in `HarborReward` delegates to harbor's own `Verifier`: uploads `tests/`, runs `test.sh`, and reports the reward as parsed from `reward.json` (first) or `reward.txt` — raw passthrough, no binarization (current datasets emit 0/1 by construction). `info["status"]` is `"success"` whenever the verifier ran; `"error"` only for verification-machinery failures.

## System Prompt

The env ships two prompts:

- `system_prompt.md` (default) — drives a structured problem-solving loop: analyze state, plan next steps, execute commands, verify results.
- `system_prompt_swe.md` — a SWE-bench-tuned variant (exposed as `SWE_SYSTEM_PROMPT_PATH`).

Benchmarks override per task via `HarborTask.system_prompt` — e.g. the `swebench-verified` evaluator stamps `system_prompt_swe.md` onto every task, so the prompt survives any custom factory.
