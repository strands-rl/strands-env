# Copyright 2025-2026 Strands RL Contributors
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

"""Evaluator for MCP-Atlas benchmark."""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Iterable

from typing_extensions import override

from strands_env.core import Action, AsyncEnvFactory, TaskContext
from strands_env.eval import EvalSample, Evaluator

from ..registry import register_eval

logger = logging.getLogger(__name__)

# 20 servers that work without API keys.
DEFAULT_SERVERS = frozenset(
    {
        "arxiv",
        "calculator",
        "cli-mcp-server",
        "clinicaltrialsgov-mcp-server",
        "context7",
        "ddg-search",
        "desktop-commander",
        "fetch",
        "filesystem",
        "git",
        "mcp-code-executor",
        "mcp-server-code-runner",
        "memory",
        "met-museum",
        "open-library",
        "osm-mcp-server",
        "pubmed",
        "weather",
        "whois",
        "wikipedia",
    }
)


class MCPAtlasTaskContext(TaskContext):
    """TaskContext with MCP-Atlas-specific fields."""

    enabled_tools: list[str]
    gtfa_claims: list[str]


@register_eval("mcp-atlas")
class MCPAtlasEvaluator(Evaluator):
    """Evaluator for MCP-Atlas benchmark."""

    benchmark_name = "mcp-atlas"

    def __init__(
        self,
        env_factory: AsyncEnvFactory,
        *,
        available_servers: frozenset[str] | None = DEFAULT_SERVERS,
        **kwargs: object,
    ):
        """Initialize a `MCPAtlasEvaluator` instance.

        Args:
            env_factory: Async factory that creates a fresh `MCPAtlasEnvironment` per sample.
            available_servers: Servers available in the container. Tasks requiring
                servers outside this set are skipped. None disables filtering.
            **kwargs: Forwarded to `Evaluator.__init__`.
        """
        super().__init__(env_factory=env_factory, **kwargs)  # type: ignore[arg-type]
        self._available_servers = available_servers

    @override
    def validate_sample(self, sample: EvalSample) -> bool:
        """Abort samples where reward is missing or judge failed, so they are retried on resume."""
        reward = sample.step_result.reward
        if reward is None:
            return False
        return reward.info.get("status") != "error"

    @override
    def load_dataset(self) -> Iterable[Action]:
        """Load MCP-Atlas tasks from HuggingFace, filter by available servers."""
        from datasets import load_dataset

        ds = load_dataset("ScaleAI/MCP-Atlas", split="train")

        actions = []
        skipped = 0
        for row in ds:
            # ENABLED_TOOLS items are plain strings or dicts with a "name" key.
            enabled_tools_raw = json.loads(row["ENABLED_TOOLS"])
            enabled_tools = [tool["name"] if isinstance(tool, dict) else tool for tool in enabled_tools_raw]
            if self._available_servers is not None:
                tool_servers = {tool_name.split("_", 1)[0] for tool_name in enabled_tools if "_" in tool_name}
                if tool_servers and not tool_servers.issubset(self._available_servers):
                    skipped += 1
                    continue

            # GTFA_CLAIMS is a Python repr string; some contain raw newlines
            # that break ast.literal_eval, so escape them first.
            gtfa_claims = ast.literal_eval(row["GTFA_CLAIMS"].replace("\n", "\\n"))

            ctx = MCPAtlasTaskContext(
                id=row["TASK"],
                enabled_tools=enabled_tools,
                gtfa_claims=gtfa_claims,
                ground_truth=row["PROMPT"],
            )
            actions.append(Action(message=row["PROMPT"], task_context=ctx))

        logger.info(
            "MCP-Atlas: loaded %d tasks (%d skipped — require unavailable servers)",
            len(actions),
            skipped,
        )
        return actions
