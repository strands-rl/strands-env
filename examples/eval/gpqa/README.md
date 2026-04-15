# GPQA

[GPQA (Graduate-Level Google-Proof Q&A)](https://huggingface.co/datasets/Idavidrein/gpqa) benchmark for expert-level multiple-choice science questions. Rule-based grading aligned with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/gpqa).

## Variants

| Name | Description |
|---|---|
| `gpqa-diamond` | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) — 198 expert-validated questions |
| `gpqa-main` | [GPQA Main](https://huggingface.co/datasets/Idavidrein/gpqa) — 448 questions |
| `gpqa-experts` | [GPQA Experts](https://huggingface.co/datasets/Idavidrein/gpqa) — expert-only validated questions |
| `gpqa-extended` | [GPQA Extended](https://huggingface.co/datasets/Idavidrein/gpqa) — 546 questions |

## Setup

The dataset is gated on Hugging Face. Accept the license at <https://huggingface.co/datasets/Idavidrein/gpqa>, then authenticate:

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

The `code_sandbox_env.py` variant requires AWS credentials with Bedrock AgentCore access.

## Files

- `chat_env.py` - Chat-only (no tools) — tests pure parametric reasoning
- `code_sandbox_env.py` - Environment hook using `CodeSandboxEnv` with Python execution

## Usage

```bash
# Chat-only
strands-env eval run gpqa-diamond \
    --env examples.eval.gpqa.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10

# With code sandbox
strands-env eval run gpqa-diamond \
    --env examples.eval.gpqa.code_sandbox_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
