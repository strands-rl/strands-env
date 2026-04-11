# IFEval

[IFEval](https://huggingface.co/datasets/google/IFEval) (Instruction-Following Evaluation) is a rule-based benchmark of 541 prompts that each carry one or more **verifiable** instructions — "write at least 3 paragraphs", "include keyword X", "respond in French", etc. Scoring is deterministic Python, not LLM-as-judge.

## Metrics

`IFEvalReward.info` exposes the canonical four metrics per sample:

| Metric | Meaning |
|---|---|
| `prompt_level_strict` | 1.0 iff **all** instructions on the prompt are followed (strict). |
| `prompt_level_loose` | Same as strict, but on responses with markdown `*` stripped + leading/trailing lines optionally removed. |
| `inst_level_strict` | Fraction of instructions followed (strict). |
| `inst_level_loose` | Fraction of instructions followed (loose). |

Scalar `reward` defaults to `prompt_level_strict_acc`. Override via `--env-config '{"ifeval_metric": "inst_level_loose_acc"}'`.

## Setup

Install the optional `[ifeval]` extra (pulls in `lm_eval[ifeval]`, which provides the vendored grader + `langdetect`, `immutabledict`, `nltk>=3.9.1`):

```bash
pip install -e ".[ifeval]"
```

The first run will auto-download NLTK's `punkt_tab` tokenizer data (~10 MB) into `~/nltk_data`. No API keys or cloud access required.

## Environments

| File | Description |
|---|---|
| `chat_env.py` | Chat-only (no tools). IFEval is graded locally, so no judge model is needed. |

## Usage

```bash
strands-env eval run ifeval \
    --env examples.eval.ifeval.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 32768 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
