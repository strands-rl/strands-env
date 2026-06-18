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

Multi-turn agent vs LLM user-sim over a shared in-memory DB. One `env.step()`
runs the full episode inside a single `agent.invoke_async()`, driven turn-by-turn
by `Tau2BenchUserSimHook` via `AfterInvocationEvent.resume` (strands-agents >= 1.30.0).
"""

from __future__ import annotations

import importlib
import logging
import re
from copy import deepcopy
from typing import Any, Literal

from strands import Agent
from strands.agent import AgentResult
from strands.agent.conversation_manager import NullConversationManager
from strands.handlers.callback_handler import PrintingCallbackHandler
from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry
from strands.telemetry.metrics import EventLoopMetrics
from typing_extensions import NotRequired, Unpack, override

from strands_env.core import Action, Environment, ModelFactory
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RewardFunction, StepResult

from .reward import Tau2BenchReward
from .tool import Tau2BenchTool

logger = logging.getLogger(__name__)

#: Canned agent opening, seeded into agent history without an LLM call.
DEFAULT_FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"
AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."
)
AGENT_SYSTEM_PROMPT_TEMPLATE = (
    "<instructions>\n{agent_instruction}\n</instructions>\n<policy>\n{domain_policy}\n</policy>"
)
USER_SYSTEM_PROMPT_TEMPLATE = "{guidelines}\n\n<scenario>\n{instructions}\n</scenario>"
AGENT_STOP_RE = re.compile(r"###STOP###")
USER_STOP_RE = re.compile(r"###(STOP|TRANSFER|OUT-OF-SCOPE)###")


class Tau2BenchConfig(EnvironmentConfig):
    """Serializable configuration for `Tau2BenchEnv`."""

    domain: Literal["airline", "retail", "telecom"]
    task: dict[str, Any]
    user_sim_guidelines: str
    max_turns: NotRequired[int]


def _extract_text(agent_result: AgentResult | None) -> str:
    """Return the text from an `AgentResult.message`, or empty string."""
    if agent_result is None:
        return ""
    return next((b["text"] for b in agent_result.message.get("content") or [] if "text" in b), "")


def build_tau2_env(domain: str, initial_db: Any, task: Any) -> Any:
    """Build a fresh tau2 domain `Environment` from `initial_db`, applying `task.initial_state` if set."""
    domain_mod = importlib.import_module(f"tau2.domains.{domain}.environment")
    tau2_env = domain_mod.get_environment(db=deepcopy(initial_db))
    if task.initial_state is not None:
        tau2_env.set_state(
            task.initial_state.initialization_data,
            task.initial_state.initialization_actions,
            [],  # no message_history resume
        )
    return tau2_env


class Tau2BenchUserSimHook(HookProvider):
    """Drive the multi-turn dialogue via `AfterInvocationEvent.resume`."""

    def __init__(self, user_sim: Agent, max_turns: int):
        """Initialize a `Tau2BenchUserSimHook` instance."""
        self.user_sim = user_sim
        self.max_turns = max_turns
        self.turn_count = 0
        self.termination: Literal["aborted", "agent_stop", "max_turns", "user_stop"] = "aborted"

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Subscribe to `AfterInvocationEvent`."""
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        self.turn_count += 1
        agent_text = _extract_text(event.result)
        if AGENT_STOP_RE.search(agent_text):
            self.termination = "agent_stop"
            return
        if self.turn_count >= self.max_turns:
            self.termination = "max_turns"
            return
        try:
            user_text = _extract_text(await self.user_sim.invoke_async(agent_text))
        except Exception as e:
            logger.warning("user-sim failed at turn %d: %s", self.turn_count, e)
            self.termination = "aborted"
            return
        if USER_STOP_RE.search(user_text):
            self.termination = "user_stop"
            # Append the terminating user msg so it shows up in `observation.messages`.
            event.agent.messages.append({"role": "user", "content": [{"text": user_text}]})
            return
        event.resume = user_text


class Tau2BenchEnv(Environment):
    """tau2-bench env: thin wrapper; multi-turn driven by `Tau2BenchUserSimHook`."""

    def __init__(
        self,
        *,
        agent_model_factory: ModelFactory,
        user_model_factory: ModelFactory,
        initial_db: Any,
        reward_fn: RewardFunction | None = None,
        judge_model_factory: ModelFactory | None = None,
        **config: Unpack[Tau2BenchConfig],
    ):
        """Initialize a `Tau2BenchEnv` instance."""
        super().__init__(model_factory=agent_model_factory, reward_fn=None, **config)  # type: ignore[misc]
        self.user_model_factory = user_model_factory
        self.initial_db = initial_db
        self.domain: Literal["airline", "retail", "telecom"] = self.config["domain"]
        self.task: dict[str, Any] = self.config["task"]
        self.max_turns: int = self.config.get("max_turns", 100)

        # Populated by `reset()`.
        self.tau2_env: Any = None
        self.user_sim: Agent | None = None
        self.user_sim_hook: Tau2BenchUserSimHook | None = None
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
        from tau2.data_model.tasks import Task  # type: ignore[import-not-found]

        task_obj = Task.model_validate(self.task)
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

        self.system_prompt = AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_instruction=AGENT_INSTRUCTION,
            domain_policy=tau2_env.policy,
        )
        self.user_sim = Agent(
            model=self.user_model_factory(),
            tools=list(self.user_tools),
            system_prompt=USER_SYSTEM_PROMPT_TEMPLATE.format(
                guidelines=self.config["user_sim_guidelines"],
                instructions=str(task_obj.user_scenario),
            ),
            conversation_manager=NullConversationManager(),
            # Mirror base `step()`'s callback handling; default `None` silences user-sim stream.
            callback_handler=PrintingCallbackHandler() if self.verbose else None,
        )
        # User-sim's reply to the canned greeting seeds the agent's first invoke;
        # subsequent turns flow through `Tau2BenchUserSimHook` via `event.resume`.
        self.first_user_msg = _extract_text(await self.user_sim.invoke_async(DEFAULT_FIRST_AGENT_MESSAGE))
        self.user_sim_hook = Tau2BenchUserSimHook(self.user_sim, self.max_turns)

    @override
    async def step(self, action: Action) -> StepResult:
        """Inject `first_user_msg` and the canned greeting history into the action."""
        action.message = self.first_user_msg
        action.task_context.conversation_history = [
            {"role": "assistant", "content": [{"text": DEFAULT_FIRST_AGENT_MESSAGE}]}
        ]
        return await super().step(action)

    @override
    def get_tools(self) -> list:
        """Return the agent-side tools."""
        return list(self.agent_tools)

    @override
    def get_hooks(self) -> list:
        """Return the multi-turn driver hook."""
        return [self.user_sim_hook] if self.user_sim_hook is not None else []

    @override
    def compute_metrics(
        self,
        event_loop_metrics: EventLoopMetrics,
        tool_parse_errors: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Agent metrics plus tau2 termination and a `user_sim` sub-dict."""
        metrics = super().compute_metrics(event_loop_metrics, tool_parse_errors)
        if self.user_sim_hook is not None:
            metrics["tau2_termination"] = self.user_sim_hook.termination
            metrics["turn_count"] = self.user_sim_hook.turn_count
        if self.user_sim is not None:
            metrics["user_sim"] = {
                "messages": list(self.user_sim.messages),
                "message_count": len(self.user_sim.messages),
                **super().compute_metrics(
                    self.user_sim.event_loop_metrics,
                    tool_parse_errors=getattr(self.user_sim.model, "tool_parse_errors", None),
                ),
            }
        return metrics
