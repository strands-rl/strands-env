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

Create a shared HTTP client via `MCPAtlasEnvironment.create_client()` and pass it to the environment.

```python
from strands_env.environments.mcp_atlas import MCPAtlasEnvironment

# Create a shared client (caller owns lifecycle — close when done)
http_client = MCPAtlasEnvironment.create_client()

# Or with custom URL / connection pool settings
http_client = MCPAtlasEnvironment.create_client(
    base_url="http://my-host:1984",
    max_connections=<max_connections>,
    max_keepalive_connections=<max_keepalive_connections>,
)

env = MCPAtlasEnvironment(
    model_factory=model_factory,
    http_client=http_client,
    reward_fn=reward_fn,
    enabled_tools=["calculator_calculate", "fetch_fetch"],
)
await env.reset()       # fetches tools and applies filtering
result = await env.step(action)
await env.cleanup()     # clears tools (does NOT close the shared client)
```

## MCPAtlasConfig

`MCPAtlasConfig` extends `EnvironmentConfig` with MCP-Atlas-specific fields.
All config fields are serializable and passed as `**kwargs` to the constructor.

| Field | Type | Description |
|---|---|---|
| `enabled_tools` | `list[str]` | Tool names to enable (omit or empty = all tools) |
| `tool_timeout` | `int` | HTTP timeout in seconds for tool and list-tools calls (default: 60) |

## TaskContext Fields

The evaluator must prepare these fields on `TaskContext` (via `MCPAtlasTaskContext`):

| Field | Type | Used by |
|---|---|---|
| `enabled_tools` | `list[str]` | env (tool filtering) |
| `gtfa_claims` | `list[str]` | reward (per-claim evaluation) |

## Reward

`MCPAtlasRewardFunction` implements per-claim LLM-as-judge evaluation following MCP-Atlas's scoring methodology. It requires a Strands `Model` as the judge, passed when constructing the reward function:

```python
from strands_env.environments.mcp_atlas import MCPAtlasRewardFunction

# judge_model is any Strands Model instance (e.g. BedrockModel, SGLangModel, LiteLLMModel)
reward_fn = MCPAtlasRewardFunction(judge_model)
```

Each GTFA claim is scored individually:
- `fulfilled` = 1.0
- `partially_fulfilled` = 0.5
- `not_fulfilled` = 0.0

The `coverage_score` is the mean across claims. Returns binary reward: 1.0 if `coverage_score >= 0.75`, 0.0 otherwise.

## Lifecycle

- **`reset()`** — POSTs `/list-tools` to the MCP-Atlas server, filters tools by `enabled_tools`, wraps them as `MCPAtlasTool` instances.
- **`step(action)`** — Runs the Strands agent with MCP tools. Each tool call POSTs to `/call-tool`.
- **`cleanup()`** — Clears the tool list. The shared HTTP client is **not** closed (the caller owns its lifecycle).
