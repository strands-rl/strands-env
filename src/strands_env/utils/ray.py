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

"""Reusable Ray actor pool for distributing `Environment.step()` across processes."""

from __future__ import annotations

import asyncio
import itertools
import logging
from argparse import Namespace
from typing import Any

import ray
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from strands_env.core.types import Action, StepResult
from strands_env.utils.loader import load_function

logger = logging.getLogger(__name__)


@ray.remote
class EnvironmentActor:
    """Remote worker that runs environment episodes in a dedicated process.

    Loads a factory builder by dotted path, reconstructs ``args`` from a plain
    dict, and calls ``builder(args, sampling_params)`` to get an
    ``AsyncEnvFactory``. The factory is built once at init and reused for all
    subsequent ``step()`` calls.

    Concurrency is controlled upstream by the caller's semaphore (e.g. SLiME's
    ``GenerateState.semaphore``), which limits how many ``generate()`` calls
    are in flight.

    Args:
        factory_builder_path: Dotted path to a callable
            ``(args, sampling_params) -> AsyncEnvFactory``.
        args_dict: ``vars(args)`` — serialized Namespace.
        sampling_params: Sampling parameters dict.
    """

    def __init__(
        self,
        factory_builder_path: str,
        args_dict: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> None:
        """Initialize an ``EnvironmentActor`` instance."""
        factory_builder = load_function(factory_builder_path)
        args = Namespace(**args_dict)
        self.env_factory = factory_builder(args, sampling_params)

    async def step(self, action_json: str) -> str:
        """Run one environment step and return the JSON-serialized StepResult.

        Args:
            action_json: JSON string from `Action.model_dump_json()`.

        Returns:
            JSON string, reconstruct via `StepResult.model_validate_json()`.
        """
        action = Action.model_validate_json(action_json)
        env = await self.env_factory(action)
        await env.reset()
        step_result = await env.step(action)
        await env.cleanup()
        return step_result.model_dump_json()


class EnvironmentActorPool:
    """Pool of `EnvironmentActor` instances distributed across Ray nodes.

    Each actor runs in its own process with a separate GIL and event loop,
    enabling true CPU parallelism for agent episodes. Follows the
    `NodeAffinitySchedulingStrategy` pattern from SLiME's distributed
    HTTP POST (``http_utils.py``).

    Args:
        factory_builder_path: Dotted path to a callable
            `(args, sampling_params) -> AsyncEnvFactory`.
        args_dict: `vars(args)` — serialized Namespace.
        sampling_params: Sampling parameters dict.
        num_actors_per_node: Number of actors per alive Ray node.
    """

    def __init__(
        self,
        factory_builder_path: str,
        args_dict: dict[str, Any],
        sampling_params: dict[str, Any],
        num_actors_per_node: int,
    ) -> None:
        """Initialize an ``EnvironmentActorPool`` instance."""
        nodes = [n for n in ray.nodes() if n.get("Alive")]
        if not nodes:
            raise RuntimeError("No alive Ray nodes for EnvironmentActor placement.")

        # Inject actor count so the factory builder can split concurrency budgets
        args_dict = {**args_dict, "num_actors": len(nodes) * num_actors_per_node}

        self.actors: list[ActorHandle] = []
        for node in nodes:
            scheduling = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            for _ in range(num_actors_per_node):
                actor = EnvironmentActor.options(  # type: ignore[attr-defined]
                    scheduling_strategy=scheduling,
                    num_cpus=0.001,
                ).remote(
                    factory_builder_path=factory_builder_path,
                    args_dict=args_dict,
                    sampling_params=sampling_params,
                )
                self.actors.append(actor)

        self.cycle = itertools.cycle(self.actors)
        logger.info(
            "Created %d EnvironmentActor(s) across %d node(s) (%d/node).",
            len(self.actors),
            len(nodes),
            num_actors_per_node,
        )

    async def step(self, action: Action) -> StepResult:
        """Run one environment step on the next available actor.

        Uses ``asyncio.to_thread(ray.get, ...)`` to avoid blocking the
        caller's event loop.
        """
        actor = next(self.cycle)
        obj_ref = actor.step.remote(action.model_dump_json())
        result_json: str = await asyncio.to_thread(ray.get, obj_ref)
        return StepResult.model_validate_json(result_json)

    def shutdown(self) -> None:
        """Kill all managed actors."""
        for actor in self.actors:
            ray.kill(actor)
        self.actors.clear()
        self.cycle = itertools.cycle([])
        logger.info("Shut down all EnvironmentActors.")
