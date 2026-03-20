# Terminal-Bench Environment

A Docker-based environment for [Terminal-Bench](https://github.com/terminal-bench/terminal-bench) task evaluation. Each task runs in an isolated Docker container where the agent executes shell commands to solve Linux administration and DevOps challenges.

## Setup

1. **Docker** — Must be installed and running:
   ```bash
   docker info  # verify Docker is available
   ```

2. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs [Harbor](https://pypi.org/project/harbor/) (`harbor>=0.1.43`) for Docker environment management.

3. **Task data** — Each task requires a directory with:
   ```
   task_dir/
   ├── Dockerfile          # or environment/ dir
   ├── tests/
   │   └── test.sh         # verification script
   └── environment/        # files copied into container
   ```

## Usage

```python
from strands_env.environments.terminal_bench import TerminalBenchEnv

env = TerminalBenchEnv(
    model_factory=model_factory,
    task_id="task-001",
    task_dir="/path/to/task",
    trial_dir="/path/to/output",
    timeout=1200,
)

await env.reset()       # Build and start Docker container
result = await env.step(action)  # action.message = task.instruction
await env.cleanup()     # Stop and delete container
```

## Tools

- **execute_command** — Execute any shell command inside the Docker container.

## Reward

Built-in `TerminalBenchReward` (binary 0/1):
1. Uploads `tests/` to the container
2. Runs `test.sh`
3. Parses `reward.txt` output — returns 1.0 if the value is >= 1, else 0.0

Supply a custom `reward_fn` to override.

## System Prompt

The agent follows a structured problem-solving loop: analyze state, plan next steps, execute commands, and verify results.
