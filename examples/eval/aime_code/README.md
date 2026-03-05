# AIME Code

AIME math competition benchmark using `CodeSandboxEnv` (AWS Bedrock AgentCore Code Interpreter).

## Variants

| Name | Description |
|---|---|
| `aime-2024` | [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| `aime-2025` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| `aime-2026` | [AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026) |

## Setup

Requires AWS credentials with Bedrock AgentCore access.

## Files

- `code_sandbox_env.py` - Environment hook using `CodeSandboxEnv` with Python execution

## Usage

```bash
strands-env eval run aime-2026 \
    --env examples/eval/aime_code/code_sandbox_env.py \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
