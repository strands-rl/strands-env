# MCP-Atlas Environment

MCP environment for [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) benchmark — 500 tasks across 36 MCP servers (220+ tools).

## Usage

Requires a running [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) server. Pass its base URL via a shared `httpx.AsyncClient`.

```python
import httpx
from strands_env.environments.mcp_atlas import MCPAtlasEnvironment

http_client = httpx.AsyncClient(base_url="http://<mcp-atlas-host>:<port>")

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

`MCPAtlasRewardFunction` implements per-claim LLM-as-judge evaluation following MCP-Atlas's scoring methodology. The judge model is constructed by the caller and passed as `reward_fn`.

Each GTFA claim is scored individually:
- `fulfilled` = 1.0
- `partially_fulfilled` = 0.5
- `not_fulfilled` = 0.0

The `coverage_score` is the mean across claims. Returns binary reward: 1.0 if `coverage_score >= 0.75`, 0.0 otherwise.

## Lifecycle

- **`reset()`** — POSTs `/list-tools` to the MCP-Atlas server, filters tools by `enabled_tools`, wraps them as `MCPAtlasTool` instances.
- **`step(action)`** — Runs the Strands agent with MCP tools. Each tool call POSTs to `/call-tool`.
- **`cleanup()`** — Clears the tool list. The shared HTTP client is **not** closed (the caller owns its lifecycle).
