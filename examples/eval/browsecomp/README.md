# BrowseComp

[BrowseComp](https://openai.com/index/browsecomp/) benchmark for browsing agents with LLM-as-judge grading.

## Environments

| File | Description |
|---|---|
| `chat_env.py` | Chat-only (no tools) — tests pure parametric knowledge |

## Setup

Requires AWS credentials with Bedrock access for the judge model.

Set `JUDGE_MODEL_ID` to override the default judge (defaults to `us.anthropic.claude-sonnet-4-20250514-v1:0`).

## Usage

```bash
strands-env eval run browsecomp \
    --env examples.eval.browsecomp.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
