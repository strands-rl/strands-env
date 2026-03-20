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

result = await env.step(action)
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

Serializable config via `WebSearchConfig` (passed as `**kwargs`):

- `search_provider` — `"serper"` (default) or `"google"`
- `search_timeout` — HTTP timeout in seconds (default 10)
- `blocked_domains` — domains to exclude from results
- `scrape_enabled` — enable web scraping (default `False`)
- `scrape_timeout` — scrape HTTP timeout (default 30)
- `scrape_token_budget` — max tokens of page content to keep (default 5000)

Non-serializable params (named args):

- `search_concurrency` — `Semaphore` or `int` for search rate limiting (default 10)
- `scrape_concurrency` — `Semaphore` or `int` for scrape rate limiting (default 10)
- `summarizer_model_factory` — model factory for LLM-based content summarization

## Reward

No built-in reward function. Supply a custom `reward_fn`.

## System Prompt

The agent is instructed to search the web, optionally scrape pages for detail, and synthesize findings into a clear, sourced answer.
