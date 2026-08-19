"""The per-sample input type for `MCPAtlasEnv`."""

from __future__ import annotations

from pydantic import Field

from strands_env.core import Task


class MCPAtlasTask(Task):
    """One MCP-Atlas sample: the tool filter and the claims the judge scores."""

    enabled_tools: list[str] = Field(description="Tool names to enable — strict filter; empty enables none.")
    gtfa_claims: list[str] = Field(description="Ground-truth final-answer claims, judged one by one.")
