# Code Sandbox Environment

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
from strands_env.environments.code_sandbox import CodeSandboxEnv
from strands_env.utils.aws import get_client

client = get_client("bedrock-agentcore", region="us-east-1")
env = CodeSandboxEnv(
    model_factory=model_factory,
    client=client,
    mode="code",  # "code", "terminal", or "code_and_terminal"
)

result = await env.step(action)
await env.cleanup()  # Clean up code interpreter session
```

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
