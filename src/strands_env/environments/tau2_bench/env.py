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

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal

from strands.telemetry.metrics import EventLoopMetrics
from typing_extensions import NotRequired, Unpack, override

from strands_env.core import Environment, ModelFactory, Task
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RewardFunction, RolloutResult, TerminationReason

from . import _tau2
from .reward import Tau2BenchReward
from .simulator import Tau2BenchTerminationReason, Tau2BenchUserSimulator
from .tool import Tau2BenchTool

if TYPE_CHECKING:
    from ._tau2 import DB, Tau2Task
    from ._tau2 import Environment as Tau2Environment

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
    task: dict[str, Any]
    max_steps: NotRequired[int]


def build_tau2_env(domain: str, initial_db: DB, task: Tau2Task) -> Tau2Environment:
    """Build a fresh tau2 domain `Environment` from `initial_db`, applying `task.initial_state` if set."""
    tau2_env = _tau2.build_environment(domain, db=deepcopy(initial_db))
    if task.initial_state is not None:
        tau2_env.set_state(
            task.initial_state.initialization_data,
            task.initial_state.initialization_actions,
            [],  # no message_history resume
        )
    return tau2_env


class Tau2BenchEnv(Environment):
    """tau2-bench env: thin wrapper; multi-turn driven by `Tau2BenchUserSimulator`."""

    def __init__(
        self,
        *,
        agent_model_factory: ModelFactory,
        user_model_factory: ModelFactory,
        initial_db: DB,
        reward_fn: RewardFunction | None = None,
        judge_model_factory: ModelFactory | None = None,
        **config: Unpack[Tau2BenchConfig],
    ):
        """Initialize a `Tau2BenchEnv` instance."""
        super().__init__(model_factory=agent_model_factory, reward_fn=None, **config)  # type: ignore[misc]
        self.user_model_factory = user_model_factory
        self.initial_db: DB = initial_db
        self.domain: Literal["airline", "retail", "telecom"] = self.config["domain"]
        self.task: dict[str, Any] = self.config["task"]
        self.max_steps: int = self.config.get("max_steps", 100)

        # Populated by `reset()`.
        self.tau2_env: Any = None
        self.user_simulator: Tau2BenchUserSimulator | None = None
        self.agent_tools: list = []
        self.user_tools: list = []
        self.first_user_msg: str = ""

        self.reward_fn = reward_fn or Tau2BenchReward(
            self,
            judge_model=judge_model_factory() if judge_model_factory else None,
        )

    @override
    async def reset(self) -> None:
        """Build per-episode tau2 environment and prime the user-sim."""
        task_obj = _tau2.Tau2Task.model_validate(self.task)
        tau2_env = build_tau2_env(self.domain, self.initial_db, task_obj)
        self.tau2_env = tau2_env
        self.agent_tools = [Tau2BenchTool(t, tau2_env, "assistant") for t in tau2_env.tools.get_tools().values()]
        self.user_tools = (
            [
                Tau2BenchTool(t, tau2_env, "user")
                for t in tau2_env.user_tools.get_tools(include=task_obj.user_tools).values()
            ]
            if tau2_env.user_tools
            else []
        )

        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=tau2_env.policy,
        )
        self.user_simulator = Tau2BenchUserSimulator(
            model=self.user_model_factory(),
            scenario=str(task_obj.user_scenario),
            tools=self.user_tools,
            max_steps=self.max_steps,
            verbose=self.verbose,
        )
        # User-sim's reply to the canned greeting seeds the agent's first invoke;
        # subsequent turns flow through `Tau2BenchUserSimulator` via `event.resume`.
        self.first_user_msg = await self.user_simulator.first_message()

    @override
    async def rollout(self, task: Task) -> RolloutResult:
        """Inject `first_user_msg` and the canned greeting history into the task."""
        task.message = self.first_user_msg
        task.context.conversation_history = [
            {"role": "assistant", "content": [{"text": Tau2BenchUserSimulator.DEFAULT_FIRST_AGENT_MESSAGE}]}
        ]
        # The user may end the dialogue already in reply to the greeting (e.g. an
        # out-of-scope scenario) — mirror tau2's USER_STOP on the first step and skip
        # the assistant agent entirely.
        if self.user_simulator is not None and self.user_simulator.termination is Tau2BenchTerminationReason.USER_STOP:
            result = RolloutResult(
                messages=[{"role": "user", "content": [{"text": self.first_user_msg}]}],
                metrics={"message_count": 1, **self.compute_metrics(EventLoopMetrics())},
                termination_reason=TerminationReason.TASK_COMPLETE,
            )
            if self.reward_fn:
                result.reward_result = await self.reward_fn.compute(task, result)
            return result
        return await super().rollout(task)

    @override
    def get_tools(self) -> list:
        """Return the agent-side tools."""
        return list(self.agent_tools)

    @override
    def get_hooks(self) -> list:
        """Return the user simulator driving the multi-turn dialogue."""
        return [self.user_simulator] if self.user_simulator is not None else []

    @override
    def compute_metrics(
        self,
        event_loop_metrics: EventLoopMetrics,
        tool_parse_errors: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Agent metrics plus tau2 termination and a `user_sim` sub-dict."""
        metrics = super().compute_metrics(event_loop_metrics, tool_parse_errors)
        if self.user_simulator is not None:
            metrics["tau2_termination"] = self.user_simulator.termination
            sim_agent = self.user_simulator.agent
            metrics["user_sim"] = {
                "messages": list(sim_agent.messages),
                "message_count": len(sim_agent.messages),
                **super().compute_metrics(
                    sim_agent.event_loop_metrics,
                    tool_parse_errors=getattr(sim_agent.model, "tool_parse_errors", None),
                ),
            }
        return metrics
