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

"""tau2-bench environment.

Multi-turn agent vs LLM user-sim over a shared in-memory DB. One `env.rollout()`
runs the full episode inside a single `agent.invoke_async()`, driven turn-by-turn
by `Tau2BenchUserSimulator` via `AfterInvocationEvent.resume` (strands-agents >= 1.30.0).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from strands.types.content import Message
from strands_sglang import LoopLimiter
from typing_extensions import NotRequired, Unpack, override

from strands_env.core import Environment, ModelFactory, Task
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RolloutResult, TerminationReason

from . import _tau2
from .reward import Tau2BenchReward
from .simulator import Tau2BenchTerminationReason, Tau2BenchUserSimulator
from .tool import Tau2BenchTool

if TYPE_CHECKING:
    from ._tau2 import Tau2Environment, Tau2Task

logger = logging.getLogger(__name__)

#: Verbatim tau2 agent system prompt (`llm_agent.py`: SYSTEM_PROMPT with AGENT_INSTRUCTION
#: inlined) — do not edit without diffing against upstream; prompt fidelity is score fidelity.
SYSTEM_PROMPT_TEMPLATE = """
<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class Tau2BenchConfig(EnvironmentConfig):
    """Serializable configuration for `Tau2BenchEnv`."""

    domain: Literal["airline", "retail", "telecom"]
    tau2_task: dict[str, Any]  # parsed via `Tau2Task.model_validate`
    max_steps: NotRequired[int]  # tau2-style step budget: total message count across agent and user-sim


class Tau2BenchEnv(Environment):
    """Tau2-bench environment with user-simulator driving the multi-turn dialogue."""

    tau2_task: Tau2Task
    tau2_env: Tau2Environment
    user_simulator: Tau2BenchUserSimulator

    def __init__(
        self,
        *,
        agent_model_factory: ModelFactory,
        user_model_factory: ModelFactory,
        judge_model_factory: ModelFactory | None = None,
        **config: Unpack[Tau2BenchConfig],
    ):
        """Initialize a `Tau2BenchEnv` instance."""
        super().__init__(model_factory=agent_model_factory, reward_fn=None, **config)  # type: ignore[misc]

        self.domain: Literal["airline", "retail", "telecom"] = self.config["domain"]
        self.max_steps: int = self.config.get("max_steps", 100)
        self.user_model_factory = user_model_factory
        self.agent_tools: list = []
        self.user_tools: list = []

        judge_model = judge_model_factory() if judge_model_factory else None
        self.reward_fn: Tau2BenchReward = Tau2BenchReward(env=self, judge_model=judge_model)

    @override
    async def reset(self, task: Task) -> None:
        """Build the tau2 environment based on the task config."""
        self.tau2_task = _tau2.Tau2Task.model_validate(self.config["tau2_task"])
        self.tau2_env = _tau2.build_task_environment(self.domain, self.tau2_task)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=self.tau2_env.policy,
        )
        self.agent_tools = [
            Tau2BenchTool(t, self.tau2_env, "assistant") for t in self.tau2_env.tools.get_tools().values()
        ]
        self.user_tools = (
            [
                Tau2BenchTool(t, self.tau2_env, "user")
                for t in self.tau2_env.user_tools.get_tools(include=self.tau2_task.user_tools).values()
            ]
            if self.tau2_env.user_tools
            else []
        )
        self.user_simulator = Tau2BenchUserSimulator(
            model=self.user_model_factory(),
            scenario=str(self.tau2_task.user_scenario),
            tools=self.user_tools,
            max_steps=self.max_steps,
            verbose=self.verbose,
        )

    @override
    async def _rollout(self, task: Task) -> RolloutResult:
        """Seed the greeting exchange into the task, then run the episode."""
        # task prompt is simulated by the user simulator
        task.message = await self.user_simulator.first_message()
        task.conversation_history = [
            Message(role="assistant", content=[{"text": Tau2BenchUserSimulator.GREETING_MESSAGE}])
        ]

        # if the user stopped the dialogue after the greeting, skip the rollout
        if self.user_simulator.termination is Tau2BenchTerminationReason.USER_STOP:
            result = RolloutResult(
                messages=[{"role": "user", "content": [{"text": task.message}]}],
                # A fresh limiter reports the truth: the agent loop processed nothing
                # `agent=None`: the user stopped at the greeting, no assistant agent ever ran
                metrics=self.compute_metrics(None, LoopLimiter()),
                termination_reason=TerminationReason.TASK_COMPLETE,
            )
            logger.warning("user ended the dialogue at greeting (%s...)", task.message[:80])
        else:
            result = await super()._rollout(task)

        # both paths report the user-simulator's side of the episode
        result.metrics["user_simulator"] = {
            "termination": self.user_simulator.termination,
            "messages": list(self.user_simulator.agent.messages),
            **self.compute_metrics(self.user_simulator.agent, self.user_simulator.limiter),
        }
        return result

    @override
    def get_tools(self) -> list:
        """Return the agent-side tools."""
        return list(self.agent_tools)

    @override
    def get_hooks(self) -> list:
        """Return the user simulator driving the multi-turn dialogue."""
        return [self.user_simulator]
