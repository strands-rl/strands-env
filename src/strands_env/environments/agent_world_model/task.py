from __future__ import annotations

from pydantic import Field

from strands_env.core import Task


class AgentWorldModelTask(Task):
    """One AgentWorldModel sample: scenario, verifier code, and the pristine DB snapshot."""

    scenario: str = Field(description="Scenario name in `gen_envs.jsonl` (which synthetic world to boot).")
    task_idx: int = Field(description="Task index within the scenario (used in logs).")
    verify_code: str = Field(description="Python source defining `verify_task_completion(...)`.")
    initial_db_path: str = Field(
        description="Pristine SQLite snapshot for this scenario; the episode's working DB is cloned from it."
    )
