# SealQA

[SealQA](https://huggingface.co/datasets/vtllms/sealqa) (SEarch-Augmented Language models QA) benchmark with LLM-as-judge grading. Evaluates reasoning over conflicting, noisy, or unhelpful search results.

## Variants

| Name | Description |
|---|---|
| `sealqa-seal-0` | [SealQA Seal-0](https://huggingface.co/datasets/vtllms/sealqa) — 111 questions |
| `sealqa-seal-hard` | [SealQA Seal-Hard](https://huggingface.co/datasets/vtllms/sealqa) — 254 questions |

## Setup

Requires AWS credentials with Bedrock access for the judge model.

Set `JUDGE_MODEL_ID` to override the default judge (defaults to `us.anthropic.claude-sonnet-4-20250514-v1:0`).

## Environments

| File | Description |
|---|---|
| `chat_env.py` | Chat-only (no tools) — tests pure parametric knowledge |

## Usage

```bash
python -m strands_env.eval \
    --benchmark sealqa-seal-0 \
    --env examples.eval.sealqa.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `python -m strands_env.eval --help` for all CLI options.
