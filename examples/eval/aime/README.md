# AIME Code

AIME math competition benchmark using `AgentCoreCodeEnv` (AWS Bedrock AgentCore Code Interpreter).

## Variants

| Name | Description |
|---|---|
| `aime-2024` | [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| `aime-2025` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) |
| `aime-2026` | [AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026) |

## Setup

Requires AWS credentials with Bedrock AgentCore access.

## Environments

- `chat_env.py` - Chat-only (no tools) — tests pure parametric knowledge
- `agentcore_code_env.py` - Environment hook using `AgentCoreCodeEnv` with Python execution

## Usage

```bash
python -m strands_env.eval \
    --benchmark aime-2026 \
    --env examples.eval.aime.agentcore_code_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
