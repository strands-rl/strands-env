# MCP-Atlas

[MCP-Atlas](https://github.com/scaleapi/mcp-atlas) benchmark — 500 tasks across 36 MCP servers with per-claim LLM-as-judge grading.

## Environments

| File | Description |
|---|---|
| `env.py` | MCP-Atlas Docker environment — agent calls tools via HTTP to a shared container |

## Setup

1. **Docker** — Start the MCP-Atlas container:
   ```bash
   # Default — 20 servers that work without API keys
   docker run -d -p 1984:1984 ghcr.io/scaleapi/mcp-atlas:1.2.5

   # All servers — fill in API keys first
   cp src/strands_env/environments/mcp_atlas/.env.template .env
   docker run -d -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.5
   ```

   Without API keys, 20 of 36 servers are available. The evaluator automatically skips tasks that require unavailable servers. See [`src/strands_env/environments/mcp_atlas/.env.template`](../../../src/strands_env/environments/mcp_atlas/.env.template) for the full list of API keys.

2. **Judge model** — Set `judge_model_id` via `--env-config` to override the default judge (defaults to Bedrock `us.anthropic.claude-sonnet-4-20250514-v1:0`). To use a different backend (e.g. Gemini via LiteLLM), override judge model construction in `env.py`.

## Usage

```bash
strands-env eval run mcp-atlas \
    --env examples.eval.mcp_atlas.env \
    --backend sglang \
    --base-url http://localhost:30000 \
    --max-tokens 16384 \
    --n-samples-per-prompt 1 \
    --max-concurrency 10
```

See `strands-env eval run --help` for all CLI options.
