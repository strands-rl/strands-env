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

"""Ray actor pool for distributing `Environment.step()` across processes."""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from strands_env.core.types import Action, RewardResult, StepResult
from strands_env.utils.loader import load_function

logger = logging.getLogger(__name__)


@ray.remote
class EnvironmentActor:
    """Remote worker that runs environment episodes in a dedicated process.

    The actor is fully generic — it loads a callable via dotted path and
    calls it with the provided kwargs to produce an `AsyncEnvFactory`.
    All domain logic (model construction, reward setup, etc.) lives in the hook.

    Args:
        env_hook_path: Dotted path to a callable that returns an `AsyncEnvFactory`.
        env_hook_config: Configuration passed to the hook callable.
    """

    def __init__(self, env_hook_path: str, env_hook_config: dict[str, Any]) -> None:
        """Initialize an `EnvironmentActor` instance."""
        env_hook = load_function(env_hook_path)
        self.env_factory = env_hook(**env_hook_config)

    async def step(self, action_json: str) -> str:
        """Run one environment step and return the JSON-serialized `StepResult`.

        Args:
            action_json: JSON string from `Action.model_dump_json()`.

        Returns:
            JSON string, reconstruct via `StepResult.model_validate_json()`.
        """
        action = Action.model_validate_json(action_json)
        env = await self.env_factory(action)
        try:
            await env.reset()
            step_result = await env.step(action)
            return step_result.model_dump_json()
        finally:
            await env.cleanup()

    async def compute_reward(self, action_json: str, step_result_json: str) -> str:
        """Recompute reward for an existing rollout without re-running the agent.

        Args:
            action_json: JSON string from `Action.model_dump_json()`.
            step_result_json: JSON string from `StepResult.model_dump_json()`.

        Returns:
            JSON string, reconstruct via `RewardResult.model_validate_json()`.
        """
        action = Action.model_validate_json(action_json)
        step_result = StepResult.model_validate_json(step_result_json)
        env = await self.env_factory(action)
        try:
            if env.reward_fn is None:
                raise ValueError("Environment has no reward function configured")
            reward_result = await env.reward_fn.compute(action=action, step_result=step_result)
            return reward_result.model_dump_json()
        finally:
            await env.cleanup()


class EnvironmentActorPool:
    """Pool of `EnvironmentActor` instances distributed across Ray nodes.

    Each actor runs in its own process with a separate GIL and event loop,
    enabling true CPU parallelism for agent episodes.

    Args:
        env_hook_path: Dotted path to a callable that returns an `AsyncEnvFactory`.
        env_hook_config: Configuration passed to the hook callable in each actor.
        n_actors_per_node: Number of actors per alive Ray node.
    """

    def __init__(
        self,
        env_hook_path: str,
        env_hook_config: dict[str, Any],
        n_actors_per_node: int,
    ) -> None:
        """Initialize an `EnvironmentActorPool` instance."""
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            raise RuntimeError("No alive Ray nodes for EnvironmentActor placement.")

        self.actors: list[ActorHandle] = []
        for node in nodes:
            scheduling = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            for _ in range(n_actors_per_node):
                actor = EnvironmentActor.options(  # type: ignore[attr-defined]
                    scheduling_strategy=scheduling,
                    num_cpus=0.001,
                ).remote(
                    env_hook_path=env_hook_path,
                    env_hook_config=env_hook_config,
                )
                self.actors.append(actor)

        self.cycle = itertools.cycle(self.actors)
        logger.info(
            "Created %d EnvironmentActor(s) across %d node(s) (%d/node).",
            len(self.actors),
            len(nodes),
            n_actors_per_node,
        )

    async def step(self, action: Action) -> StepResult:
        """Run one environment step on the next available actor.

        Uses `asyncio.to_thread(ray.get, ...)` to avoid blocking the
        caller's event loop.
        """
        actor = next(self.cycle)
        obj_ref = actor.step.remote(action.model_dump_json())
        result_json: str = await asyncio.to_thread(ray.get, obj_ref)
        return StepResult.model_validate_json(result_json)

    async def compute_reward(self, action: Action, step_result: StepResult) -> RewardResult:
        """Recompute reward for an existing rollout on the next available actor."""
        actor = next(self.cycle)
        obj_ref = actor.compute_reward.remote(action.model_dump_json(), step_result.model_dump_json())
        result_json: str = await asyncio.to_thread(ray.get, obj_ref)
        return RewardResult.model_validate_json(result_json)

    def shutdown(self) -> None:
        """Kill all managed actors."""
        for actor in self.actors:
            ray.kill(actor)
        self.actors.clear()
        self.cycle = itertools.cycle([])
        logger.info("Shut down all EnvironmentActors.")
