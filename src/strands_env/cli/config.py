# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration dataclasses for CLI."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SamplingConfig:
    """Sampling parameters for model generation."""

    temperature: float | None = None
    max_new_tokens: int = 16384
    top_p: float | None = None
    top_k: int | None = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding `None` values so the model uses its own defaults."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}


@dataclass
class ModelConfig:
    """Model configuration."""

    backend: Literal["sglang", "bedrock", "kimi"] = "sglang"

    # SGLang
    base_url: str = "http://localhost:30000"
    tokenizer_path: str | None = None  # Auto-detected if None
    tool_parser: str | None = None  # Parser name or path to hook file

    # Bedrock
    model_id: str | None = None
    region: str | None = None
    profile_name: str | None = None  # AWS profile name
    role_arn: str | None = None  # For role assumption

    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = dataclasses.asdict(self)
        d["sampling"] = self.sampling.to_dict()
        return d


@dataclass
class EnvConfig:
    """Environment configuration."""

    system_prompt_path: Path | None = None
    max_tool_iters: int | None = None
    max_tool_calls: int | None = None
    max_parallel_tool_calls: int | None = None
    verbose: bool = False

    @property
    def system_prompt(self) -> str | None:
        """Load system prompt from file if path is set."""
        if self.system_prompt_path is None:
            return None
        return self.system_prompt_path.read_text()

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = dataclasses.asdict(self)
        d["system_prompt_path"] = str(self.system_prompt_path) if self.system_prompt_path else None
        d["system_prompt"] = self.system_prompt  # Save actual content for reproducibility
        return d


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    n_samples_per_prompt: int = 1
    max_concurrency: int = 10
    output_dir: Path | None = None  # Defaults to {benchmark}_eval/
    save_interval: int = 10
    keep_tokens: bool = False

    def get_output_dir(self, benchmark_name: str) -> Path:
        """Get output directory, using default if not set."""
        if self.output_dir is not None:
            return self.output_dir
        return Path(f"{benchmark_name}_eval")

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        d = dataclasses.asdict(self)
        d["output_dir"] = str(self.output_dir) if self.output_dir else None
        return d
