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

"""Utilities for `SGLang` client."""

from __future__ import annotations

import httpx


def check_server_health(base_url: str, timeout: float = 5.0) -> None:
    """Check if the SGLang server is reachable.

    Notes:
        Sync convenience using httpx. For async runtime use, see
        `SGLangClient.health()` which uses aiohttp.
    """
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise ConnectionError(f"SGLang server at {base_url} is not reachable: {e}") from e


def get_model_id(base_url: str, timeout: float = 5.0) -> str:
    """Get the model ID from the SGLang server via ``/v1/models``."""
    response = httpx.get(f"{base_url}/v1/models", timeout=timeout)
    response.raise_for_status()
    models = response.json()["data"]
    if not models:
        raise RuntimeError(f"No models found at {base_url}/v1/models")
    return str(models[0]["id"])
