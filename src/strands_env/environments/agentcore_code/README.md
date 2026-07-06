# AgentCore Code Environment

A sandboxed code execution environment using AWS Bedrock AgentCore Code Interpreter. Supports Python execution, shell commands, or both.

## Setup

1. **AWS credentials** — Configure AWS credentials with access to Bedrock AgentCore:
   ```bash
   aws configure
   # or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
   ```

2. **No additional pip dependencies** — uses `boto3` which is included in the base `strands-env` install.

## Usage

```python
from strands_env.environments.agentcore_code import AgentCoreCodeEnv
from strands_env.utils.aws import get_client

client = get_client("bedrock-agentcore", region="us-east-1")
env = AgentCoreCodeEnv(
    model_factory=model_factory,
    client=client,
    mode="code",  # "code", "terminal", or "code_and_terminal"
)

result = await env.rollout(task)  # reset -> episode -> reward -> cleanup (session closed)
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `mode` | `"code"` | Which tools to expose: `"code"`, `"terminal"`, or `"code_and_terminal"` |
| `session_timeout_seconds` | `3600` | Sandbox session lifetime |

Base knobs come from `EnvironmentConfig`. Non-serializable named args: `client` (boto3 `bedrock-agentcore` client; default created on demand), `quotas` (share one `CodeInterpreterQuotas` across envs for account-wide limits).

## Tools

Depends on the configured mode:

| Mode | Tools |
|---|---|
| `"code"` | `execute_code` (Python) |
| `"terminal"` | `execute_command` (shell) |
| `"code_and_terminal"` | Both |

## Reward

No built-in reward function. Supply a custom `reward_fn`.

## System Prompt

The agent is instructed to write and execute code to solve problems, breaking tasks into smaller steps and verifying results.
