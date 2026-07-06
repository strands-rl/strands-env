# Web Search Environment

A web search environment that gives the agent search and optional web scraping tools. Supports Serper and Google Custom Search providers.

## Setup

Set API credentials as environment variables depending on your search provider:

```bash
# Serper (default)
export SERPER_API_KEY="your-key"

# Google Custom Search
export GOOGLE_API_KEY="your-key"
export GOOGLE_CSE_ID="your-cse-id"
```

## Usage

```python
from strands_env.environments.web_search import WebSearchEnv

# Search only (default)
env = WebSearchEnv(model_factory=model_factory)

# Search + scrape
env = WebSearchEnv(model_factory=model_factory, scrape_enabled=True)

# Search + scrape with LLM summarization
env = WebSearchEnv(
    model_factory=model_factory,
    scrape_enabled=True,
    summarizer_model_factory=summarizer_factory,
)

result = await env.rollout(task)
await env.cleanup()  # close shared HTTP sessions once, after ALL rollouts (sessions are reused across episodes)
```

## Tools

Depends on configuration:

| Config | Tools |
|---|---|
| Default | `serper_search` |
| `search_provider="google"` | `google_search` |
| `scrape_enabled=True` | search + `scrape` |
| `scrape_enabled=True` + `summarizer_model_factory` | search + `scrape_and_summarize` |

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `search_provider` | `"serper"` | `"serper"` or `"google"` |
| `search_timeout` | `10` | Seconds per search API call |
| `blocked_domains` | — | Appended to queries as `-site:` exclusions |
| `scrape_enabled` | `False` | Expose the scrape tool |
| `scrape_timeout` | `50` | Seconds per page fetch |
| `scrape_token_budget` | `20000` | Max page tokens kept for summarization |

Base knobs come from `EnvironmentConfig`. Non-serializable named args: `search_concurrency` / `scrape_concurrency` (a shared `Semaphore` = one budget across envs, an `int` = per-env cap), `summarizer_model_factory` (without it the scrape tool returns raw page content).

## Reward

No built-in reward function. Supply a custom `reward_fn`.

## System Prompt

The agent is instructed to search the web, optionally scrape pages for detail, and synthesize findings into a clear, sourced answer.
