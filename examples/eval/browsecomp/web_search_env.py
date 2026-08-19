"""Example environment hook for BrowseComp evaluation with Serper search + Jina-based web scraping."""

import asyncio

from strands_env.core.models import bedrock_model_factory, build_model_factory
from strands_env.environments.web_search.env import WebSearchEnv
from strands_env.eval.benchmarks.browsecomp import BrowseCompReward
from strands_env.utils.aws import get_session

SEARCH_SEMAPHORE = asyncio.Semaphore(30)
SCRAPE_SEMAPHORE = asyncio.Semaphore(30)


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `WebSearchEnv`."""
    model_factory = build_model_factory(model_config)
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(
            region_name="us-west-2", profile_name=profile_name, role_arn=env_config.get("judge_model_role_arn")
        )
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    reward_fn = BrowseCompReward(judge_model=judge_models)

    async def env_factory():
        return WebSearchEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            summarizer_model_factory=model_factory,
            search_provider=env_config.get("search_provider", "serper"),
            scrape_enabled=env_config.get("scrape_enabled", True),
            scrape_token_budget=env_config.get("scrape_token_budget", 20000),
            scrape_timeout=env_config.get("scrape_timeout", 50),
            search_concurrency=SEARCH_SEMAPHORE,
            scrape_concurrency=SCRAPE_SEMAPHORE,
        )

    return env_factory
