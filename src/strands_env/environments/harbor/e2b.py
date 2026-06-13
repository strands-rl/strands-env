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

"""Prebaked-template adapter for the Harbor `e2b` backend.

Self-hosted e2b clusters may not implement Harbor's auto-build route, so
`PrebakedE2BEnvironment` boots from a pre-baked `template_id` (resolved via a
`templates.json` mapping or `E2B_TEMPLATES_PATH`) instead of building one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import e2b.api as _e2b_api
from e2b.exceptions import AuthenticationException
from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.paths import EnvironmentPaths
from typing_extensions import override

if TYPE_CHECKING:
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

    from .env import PrebakedE2BConfig


class PrebakedE2BEnvironment(E2BEnvironment):
    """E2BEnvironment that boots from a pre-baked template, skipping Harbor's auto-build.

    Identical to `E2BEnvironment` plus a keyword-only `template_id` pinning the
    already-baked template to boot from. Retries are inherited from the base.
    """

    def __init__(self, *args: Any, template_id: str, **kwargs: Any) -> None:
        """Initialize a `PrebakedE2BEnvironment` instance.

        `template_id` (keyword-only) pins the pre-baked template; the rest is
        forwarded to `E2BEnvironment.__init__`.
        """
        super().__init__(*args, **kwargs)
        self._template_id = template_id

    @classmethod
    def from_config(
        cls,
        config: PrebakedE2BConfig,
        *,
        task_id: str,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: TaskEnvironmentConfig,
    ) -> PrebakedE2BEnvironment:
        """Build a `PrebakedE2BEnvironment` from a `PrebakedE2BConfig`.

        Applies the connection env vars the e2b SDK reads, resolves the template
        id (explicit or via the templates map), and constructs the environment.
        """
        cls._apply_connection_env(config)
        template_id = config.get("template_id") or cls.resolve_template_id(
            task_name=task_id,
            template_map_path=config.get("templates_json"),
        )
        return cls(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            template_id=template_id,
        )

    @override
    async def start(self, force_build: bool) -> None:
        # Skip the build path; boot from the pinned template id.
        self._template_name = self._template_id
        if force_build:
            self.logger.warning(
                "force_build=True requested but PrebakedE2BEnvironment ignores it. "
                "Re-bake the template against the e2b cluster if needed.",
            )
        await self._create_sandbox()
        if not self._sandbox:
            raise RuntimeError("Sandbox not found but was just created.")
        # The build path we skip is what bakes Harbor's log/test dirs into the
        # image, so create them here (reward.py reads/writes under /logs/verifier).
        # Mount targets are unioned in so user-declared volumes still work.
        harbor_dirs = [
            EnvironmentPaths.logs_dir,
            EnvironmentPaths.agent_dir,
            EnvironmentPaths.verifier_dir,
            EnvironmentPaths.artifacts_dir,
            EnvironmentPaths.tests_dir,
        ]
        mount_targets = self._mount_targets(writable_only=True)
        await self.ensure_dirs([*harbor_dirs, *mount_targets])
        # Base behaviour: copy a prebuilt-image task's environment/ into the
        # workdir. Self-guards to a no-op when not needed, so always safe.
        await self._upload_environment_dir_after_start()

    @override
    async def _create_template(self) -> None:  # type: ignore[override]
        # Defensive: start() skips the build path, so this should be unreachable.
        raise RuntimeError(
            "PrebakedE2BEnvironment does not auto-build templates. "
            "Templates must be baked against the e2b cluster before eval.",
        )

    @staticmethod
    def _apply_connection_env(config: PrebakedE2BConfig) -> None:
        """Write `domain`/`api_key` into the env vars harbor's E2BEnvironment + SDK read."""
        if domain := config.get("domain"):
            os.environ["E2B_DOMAIN"] = domain
        if api_key := config.get("api_key"):
            os.environ["E2B_API_KEY"] = api_key
        elif api_key_file := config.get("api_key_file"):
            key = Path(api_key_file).expanduser().read_text().strip()
            if not key:
                raise ValueError(f"e2b api_key_file is empty: {api_key_file}")
            os.environ["E2B_API_KEY"] = key

    @staticmethod
    def _permissive_validate_api_key(api_key: str) -> None:
        """Validate an e2b API key by `e2b_` prefix + non-empty body only.

        The SDK normally requires hex keys; self-hosted clusters mint `[a-z0-9]`
        ones. Installed onto `e2b.api.validate_api_key` at import (bottom of file).
        """
        if not isinstance(api_key, str) or not api_key.startswith("e2b_") or len(api_key) <= 4:
            raise AuthenticationException('Invalid API key format: expected "e2b_" followed by a non-empty body.')

    @staticmethod
    def _load_template_map(path: str | Path) -> dict[str, str]:
        """Load a {task_name: template_id} mapping from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Templates file not found at {p}. Bake the task templates against the e2b cluster first.",
            )
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{p} must contain a JSON object mapping task_name -> template_id.")
        return data

    @classmethod
    def resolve_template_id(cls, task_name: str, template_map_path: str | Path | None = None) -> str:
        """Resolve a Harbor task name to its e2b template id.

        Args:
            task_name: The task's name (e.g. `"fix-git"`), matching the dataset
                directory and `Task.name` after Harbor's mapper has run.
            template_map_path: Path to `templates.json`; falls back to
                `E2B_TEMPLATES_PATH` when None.

        Raises:
            FileNotFoundError: If the templates file does not exist.
            KeyError: If the task has no entry (its template isn't baked yet).
        """
        if template_map_path is None:
            template_map_path = os.environ.get("E2B_TEMPLATES_PATH")
        if template_map_path is None:
            raise RuntimeError(
                "E2B_TEMPLATES_PATH not set and no template_map_path provided. "
                "Set it to a templates.json mapping task_name -> template_id.",
            )
        mapping = cls._load_template_map(template_map_path)
        # Accept both Harbor's `<benchmark>/<task>` form and the bare dir name:
        # the mapping is keyed by bare name; the evaluator passes the full Task.name.
        candidates = [task_name, task_name.split("/")[-1]]
        for candidate in candidates:
            if candidate in mapping:
                return mapping[candidate]
        raise KeyError(
            f"Task {task_name!r} has no template. "
            f"Tried: {candidates}. "
            f"Known tasks: {sorted(mapping.keys())[:10]}{'...' if len(mapping) > 10 else ''}. "
            f"Bake the template for {candidates[-1]!r} first.",
        )


# Relax the SDK's hex-only key check before any Sandbox is constructed; some
# self-hosted clusters use a broader key alphabet. See `_permissive_validate_api_key`.
_e2b_api.validate_api_key = PrebakedE2BEnvironment._permissive_validate_api_key
