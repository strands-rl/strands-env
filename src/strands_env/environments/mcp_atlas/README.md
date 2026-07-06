# MCP-Atlas Environment

MCP environment for [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) benchmark — 500 tasks across 36 MCP servers (307 tools).

## Setup

Start the [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) Docker container:

```bash
# Default — 20 servers that work without API keys
docker run -d -p 1984:1984 ghcr.io/scaleapi/mcp-atlas:1.2.5

# All servers — copy .env.template to .env and fill in API keys
docker run -d -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.5
```

## Usage

Create a shared HTTP client via `MCPAtlasEnv.create_client()` and pass it to the environment.

```python
from strands_env.environments.mcp_atlas import MCPAtlasEnv, MCPAtlasTask

# Create a shared client (caller owns lifecycle — close when done)
http_client = MCPAtlasEnv.create_client()  # or create_client(base_url=..., max_connections=...)

env = MCPAtlasEnv(model_factory=model_factory, http_client=http_client, reward_fn=reward_fn)

task = MCPAtlasTask(
    message=prompt,
    enabled_tools=["calculator_calculate", "fetch_fetch"],
    gtfa_claims=claims,
)
result = await env.rollout(task)  # reset (fetch + filter tools) -> episode -> reward -> cleanup
await http_client.aclose()        # once, after ALL rollouts
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `tool_timeout` | `60` | HTTP timeout in seconds for tool and list-tools calls |

Base knobs come from `EnvironmentConfig`. Non-serializable named args: `http_client` (shared, caller-owned), `cached_tools` (pre-fetched `/list-tools` response — share one across envs to skip per-episode refetch).

## Task Fields

| Field | Meaning |
|---|---|
| `enabled_tools` | Tool names to enable (strict filter — empty enables none) |
| `gtfa_claims` | Ground-truth final-answer claims for the per-claim judge reward |

## Reward

`MCPAtlasReward` implements per-claim LLM-as-judge evaluation following MCP-Atlas's scoring methodology. It requires a Strands `Model` as the judge, passed when constructing the reward function:

```python
from strands_env.environments.mcp_atlas import MCPAtlasReward

# judge_model is any Strands Model instance (e.g. BedrockModel, SGLangModel, LiteLLMModel)
reward_fn = MCPAtlasReward(judge_model)
```

Each GTFA claim is scored individually:
- `fulfilled` = 1.0
- `partially_fulfilled` = 0.5
- `not_fulfilled` = 0.0

The `coverage_score` is the mean across claims. Returns binary reward: 1.0 if `coverage_score >= 0.75`, 0.0 otherwise.

## Lifecycle

- **`reset(task)`** — POSTs `/list-tools` to the MCP-Atlas server (or uses `cached_tools`), filters by `task.enabled_tools`, wraps them as `MCPAtlasTool` instances.
- **`rollout(task)`** — Runs the Strands agent with MCP tools. Each tool call POSTs to `/call-tool`.
- **`cleanup()`** — Clears the tool list. The shared HTTP client is **not** closed (the caller owns its lifecycle).
