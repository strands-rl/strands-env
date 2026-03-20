# HMMT

[HMMT](https://www.hmmt.org/) (Harvard-MIT Mathematics Tournament) benchmark using MathArena datasets.

## Variants

| Name | Description |
|---|---|
| `hmmt-feb-2025` | [HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) |
| `hmmt-nov-2025` | [HMMT November 2025](https://huggingface.co/datasets/MathArena/hmmt_nov_2025) |
| `hmmt-feb-2026` | [HMMT February 2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026) |

## Setup

No additional setup required for chat-only evaluation. Code sandbox requires AWS credentials with Bedrock AgentCore access.

## Files

- `chat_env.py` - Chat-only (no tools) — tests pure parametric knowledge
- `code_sandbox_env.py` - Environment hook using `CodeSandboxEnv` with Python execution

## Usage

```bash
strands-env eval run hmmt-feb-2026 \
    --env examples.eval.hmmt.code_sandbox_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
