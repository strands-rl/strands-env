# HLE-Verified

[HLE-Verified](https://huggingface.co/datasets/skylenage-ai/HLE-Verified) — a re-validated subset of Humanity's Last Exam. This example targets the **Gold subset** (668 fully validated items) with LLM-as-judge grading using the official HLE grader prompt.

## Benchmarks

| Name | Samples | Includes images |
|---|---|---|
| `hle-verified-gold` | 668 | yes (93 multimodal) |
| `hle-verified-gold-text` | 575 | no |

The multimodal variant attaches decoded image bytes as a Strands `ContentBlock`, so the backend model must support image inputs.

## Environments

| File | Description |
|---|---|
| `chat_env.py` | Chat-only (no tools) — tests pure parametric knowledge |

## Setup

Requires AWS credentials with Bedrock access for the judge model.

Override the judge via `--env-config '{"judge_model_id": "..."}'` (defaults to `us.anthropic.claude-sonnet-4-20250514-v1:0`).

## Usage

```bash
# Text-only subset (works with any text model)
strands-env eval run hle-verified-gold-text \
    --env examples.eval.hle_verified.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10

# Full Gold subset including multimodal samples (needs a vision-capable model)
strands-env eval run hle-verified-gold \
    --env examples.eval.hle_verified.chat_env \
    --backend bedrock \
    --model-id us.anthropic.claude-sonnet-4-20250514-v1:0 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
