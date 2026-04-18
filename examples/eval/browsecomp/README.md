# BrowseComp

[BrowseComp](https://openai.com/index/browsecomp/) benchmark for browsing agents with LLM-as-judge grading.

## Environments

| File | Description |
|---|---|
| `chat_env.py` | Chat-only (no tools) — tests pure parametric knowledge |
| `web_search_env.py` | Serper search + Jina-backed page scraping — reproduces [OpenSeeker](https://github.com/rui-ye/OpenSeeker)'s BrowseComp setup |

## Setup

All environments require AWS credentials with Bedrock access for the judge model (override with `--env-config '{"judge_model_id": "..."}'` or rely on the default `us.anthropic.claude-sonnet-4-20250514-v1:0`).

`web_search_env.py` additionally requires:

- `SERPER_API_KEY` — [Serper](https://serper.dev) search API.
- `JINA_API_KEY` — [Jina Reader](https://jina.ai/reader/) API (enforced by the scrape tool's `@requires_env` guard; a valid key is required for non-rate-limited access).

## Usage

**Chat-only (parametric-knowledge baseline):**

```bash
strands-env eval run browsecomp \
    --env examples.eval.browsecomp.chat_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

**WebSearchEnv (Serper + Jina):**

```bash
export SERPER_API_KEY=...
export JINA_API_KEY=...

strands-env eval run browsecomp \
    --env examples.eval.browsecomp.web_search_env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --tool-parser hermes \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
