# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example environment hook for BrowseComp evaluation with Serper search + Jina visit.

Uses the existing ``WebSearchToolkit`` (Serper) for search and the new
``OpenSeekerVisitToolkit`` (Jina Reader + LLM summarization) for page reading.

Required environment variables:

- ``SERPER_API_KEY`` for Serper search
- ``JINA_API_KEY`` (optional but recommended for higher Jina rate limits)

Usage::

    strands-env eval run browsecomp \\
        --env examples.eval.browsecomp.openseeker_env \\
        --backend sglang --base-url http://localhost:30000 \\
        --tool-parser hermes --max-tokens 16384 \\
        --max-concurrency 10
"""

from typing import Any

from strands_env.core import Environment
from strands_env.core.models import ModelFactory, bedrock_model_factory, build_model_factory
from strands_env.eval.benchmarks.browsecomp import BrowseCompReward
from strands_env.tools.openseeker_visit import OpenSeekerVisitToolkit
from strands_env.tools.web_search import WebSearchToolkit
from strands_env.utils.aws import get_session


class _SearchVisitEnv(Environment):
    """Thin env wiring a search tool and a visit tool into the agent."""

    def __init__(self, *, search_tool, visit_tool, cleanup_cbs, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._tools = [t for t in [search_tool, visit_tool] if t is not None]
        self._cleanup_cbs = list(cleanup_cbs)

    def get_tools(self):  # type: ignore[override]
        return list(self._tools)

    async def cleanup(self):
        for cb in self._cleanup_cbs:
            try:
                await cb()
            except Exception:
                pass


def create_env_factory(model_config: dict[str, Any], **env_config: Any):
    """Create env_factory for BrowseComp evaluation with Serper search + Jina visit."""
    model_factory: ModelFactory = build_model_factory(model_config)

    # ---- Judge setup (same pattern as chat_env.py) ----
    judge_models = []
    for profile_name in env_config.get("judge_model_profiles", [None]):
        boto_session = get_session(
            region="us-west-2",
            profile_name=profile_name,
            role_arn=env_config.get("judge_model_role_arn"),
        )
        judge_models.append(
            bedrock_model_factory(
                model_id=env_config.get("judge_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
                boto_session=boto_session,
                sampling_params={"max_new_tokens": 1024},
            )()
        )
    reward_fn = BrowseCompReward(
        judge_model=judge_models,
        max_model_retries=env_config.get("max_judge_retries", 3),
    )

    # ---- Toolkits ----
    search_toolkit = WebSearchToolkit()
    visit_toolkit = OpenSeekerVisitToolkit(summarizer_model_factory=model_factory)

    async def env_factory(_action):  # type: ignore[no-untyped-def]
        return _SearchVisitEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            search_tool=search_toolkit.serper_search,
            visit_tool=visit_toolkit.visit,
            cleanup_cbs=[search_toolkit.cleanup, visit_toolkit.cleanup],
        )

    return env_factory
