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

"""The per-sample input type for `MCPAtlasEnv`."""

from __future__ import annotations

from pydantic import Field

from strands_env.core import Task


class MCPAtlasTask(Task):
    """One MCP-Atlas sample: the tool filter and the claims the judge scores."""

    enabled_tools: list[str] = Field(description="Tool names to enable — strict filter; empty enables none.")
    gtfa_claims: list[str] = Field(description="Ground-truth final-answer claims, judged one by one.")
