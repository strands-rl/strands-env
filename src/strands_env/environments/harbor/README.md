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
from strands_env.environments.harbor import HarborEnv

env = HarborEnv(
    model_factory=model_factory,
    task_id="task-001",
    task_dir="/path/to/task",
    trial_dir="/path/to/output",
    timeout=1200,
)

await env.reset()       # Build and start container
result = await env.step(action)  # action.message = task.instruction
await env.cleanup()     # Stop and delete container
```

## Tools

- **execute_command** — Execute any shell command inside the container.

## Reward

Built-in `HarborReward` (binary 0/1):
1. Uploads `tests/` to the container
2. Runs `test.sh`
3. Parses `reward.txt` output — returns 1.0 if the value is >= 1, else 0.0

Supply a custom `reward_fn` to override.

## System Prompt

The env ships two prompts:

- `system_prompt.md` (default) — drives a structured problem-solving loop: analyze state, plan next steps, execute commands, verify results.
- `system_prompt_swe.md` — a SWE-bench-tuned variant (exposed as `SWE_SYSTEM_PROMPT_PATH`).

Benchmarks override the default via the serializable `system_prompt` config field — e.g. the `swebench-verified` evaluator injects `system_prompt_swe.md`.
