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

"""Ray actor pool for distributing `Environment.rollout()` across processes."""

from __future__ import annotations

import asyncio
import itertools
import logging
import subprocess
from typing import Any

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from strands_env.utils.loader import load_function

from .types import RewardResult, RolloutResult, Task

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
        env_hook = load_function(env_hook_path)
        self.env_factory = env_hook(**env_hook_config)

    async def rollout(self, task_json: str) -> str:
        """Run one rollout and return the JSON-serialized `RolloutResult`.

        Args:
            task_json: JSON string from `Task.model_dump_json()`.

        Returns:
            JSON string, reconstruct via `RolloutResult.model_validate_json()`.
        """
        env = await self.env_factory()
        task = env.task_cls.model_validate_json(task_json)  # restore the env's typed task from the wire
        result = await env.rollout(task)
        return result.model_dump_json()

    async def compute_reward(self, task_json: str, result_json: str) -> str:
        """Recompute reward for an existing rollout without re-running the agent.

        Args:
            task_json: JSON string from `Task.model_dump_json()`.
            result_json: JSON string from `RolloutResult.model_dump_json()`.

        Returns:
            JSON string, reconstruct via `RewardResult.model_validate_json()`.
        """
        result = RolloutResult.model_validate_json(result_json)
        env = await self.env_factory()
        task = env.task_cls.model_validate_json(task_json)  # restore the env's typed task from the wire
        try:
            if env.reward_fn is None:
                raise ValueError("Environment has no reward function configured")
            reward_result = await env.reward_fn.compute(task, result)
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

    async def rollout(self, task: Task) -> RolloutResult:
        """Run one rollout on the next available actor.

        Uses `asyncio.to_thread(ray.get, ...)` to avoid blocking the
        caller's event loop.
        """
        actor = next(self.cycle)
        obj_ref = actor.rollout.remote(task.model_dump_json())
        result_json: str = await asyncio.to_thread(ray.get, obj_ref)
        return RolloutResult.model_validate_json(result_json)

    async def compute_reward(self, task: Task, result: RolloutResult) -> RewardResult:
        """Recompute reward for an existing rollout on the next available actor."""
        actor = next(self.cycle)
        obj_ref = actor.compute_reward.remote(task.model_dump_json(), result.model_dump_json())
        result_json: str = await asyncio.to_thread(ray.get, obj_ref)
        return RewardResult.model_validate_json(result_json)

    def shutdown(self) -> None:
        """Tear down the Ray cluster on every node."""
        self.actors.clear()
        self.cycle = itertools.cycle([])

        @ray.remote(num_cpus=0)  # type: ignore[untyped-decorator]
        def ray_stop() -> None:
            subprocess.Popen(
                "(setsid ray stop --force </dev/null >/dev/null 2>&1 &)",
                shell=True,
            )

        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            strategy = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ray_stop.options(scheduling_strategy=strategy).remote()

        ray.shutdown()
        logger.info("Shut down Ray cluster across all nodes.")
