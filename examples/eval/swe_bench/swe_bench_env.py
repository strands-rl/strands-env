from __future__ import annotations

from strands_env.core.models import build_model_factory
from strands_env.environments.harbor import HarborEnv


def create_env_factory(model_config: dict, **env_config):
    """Create env_factory for `HarborEnv`."""
    model_factory = build_model_factory(model_config)

    async def env_factory() -> HarborEnv:
        """Create a new HarborEnv with its own container/pod. The sample itself arrives via `rollout(task)`."""
        return HarborEnv(model_factory=model_factory, **env_config)

    return env_factory
