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
await env.cleanup()  # Close HTTP sessions
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

`WebSearchConfig` keys (passed as `**kwargs`):

| Field | Default | Meaning |
|---|---|---|
| `search_provider` | `"serper"` | Search API provider: `"serper"` or `"google"` |
| `search_timeout` | `10` | Search HTTP timeout in seconds |
| `blocked_domains` | `None` | Domains to exclude from search results |
| `scrape_enabled` | `False` | Enable the scrape tool |
| `scrape_timeout` | `50` | Scrape HTTP timeout in seconds |
| `scrape_token_budget` | `20000` | Max tokens of scraped page content to keep |

Base knobs (`system_prompt`, `max_tool_iters`, `max_tool_calls`, `max_parallel_tool_calls`, `max_messages`, `trace_attributes`, `agent_name`, `verbose`) come from `EnvironmentConfig`.

Non-serializable params (named args):

- `search_concurrency` — `Semaphore` or `int` for search rate limiting (default 10)
- `scrape_concurrency` — `Semaphore` or `int` for scrape rate limiting (default 10)
- `summarizer_model_factory` — model factory for LLM-based content summarization

## Reward

No built-in reward function. Supply a custom `reward_fn`.

## System Prompt

The agent is instructed to search the web, optionally scrape pages for detail, and synthesize findings into a clear, sourced answer.
