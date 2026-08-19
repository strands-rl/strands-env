"""Harbor task environment for Docker/e2b-based task evaluation."""

from pathlib import Path

from .env import HarborConfig, HarborEnv
from .reward import HarborReward
from .task import HarborTask

#: SWE-bench-tuned system prompt shipped with the env; benchmarks inject it via the
#: serializable `system_prompt` config key (e.g. the `swebench-verified` evaluator).
SWE_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt_swe.md"

__all__ = ["SWE_SYSTEM_PROMPT_PATH", "HarborConfig", "HarborEnv", "HarborReward", "HarborTask"]
