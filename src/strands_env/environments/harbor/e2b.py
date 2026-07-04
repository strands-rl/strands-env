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
from typing import TYPE_CHECKING, Any, override

import e2b.api as _e2b_api
from e2b.exceptions import AuthenticationException
from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.paths import EnvironmentPaths

if TYPE_CHECKING:
    from .env import PrebakedE2BConfig


class PrebakedE2BEnvironment(E2BEnvironment):
    """E2BEnvironment that boots from a pre-baked template, skipping Harbor's auto-build.

    Identical to `E2BEnvironment` plus a keyword-only `template_id` pinning the
    already-baked template to boot from. Retries are inherited from the base.
    """

    def __init__(self, task_id: str, prebaked_e2b_config: PrebakedE2BConfig, *args: Any, **kwargs: Any) -> None:
        """Initialize a `PrebakedE2BEnvironment` instance.

        Resolves the pinned template id from `prebaked_e2b_config` (an explicit
        `template_id`, else a `templates.json` lookup by `task_id`) and exports the
        e2b connection env vars; `*args`/`**kwargs` forward to `E2BEnvironment.__init__`.
        """
        self.task_id = task_id
        self._config = prebaked_e2b_config
        self._template_id = self._config.get("template_id") or self._resolve_template_id()
        self._set_e2b_env_vars()
        super().__init__(*args, **kwargs)

    def _resolve_template_id(self) -> str:
        """Resolve a Harbor task name to its e2b template id."""
        templates_path = self._config.get("templates_json") or os.environ.get("E2B_TEMPLATES_PATH")
        if templates_path is None:
            raise RuntimeError("E2B_TEMPLATES_PATH not set and no templates_path provided.")
        templates = json.loads(Path(templates_path).read_text())
        # Accept both Harbor's `<benchmark>/<task>` form and the bare dir name.
        candidates = [self.task_id, self.task_id.split("/")[-1]]
        for name in candidates:
            if name in templates:
                return templates[name]
        raise KeyError(f"Task {self.task_id!r} has no matching template; tried: {candidates}.")

    def _set_e2b_env_vars(self) -> None:
        """Write `domain`/`api_key` into the env vars harbor's E2BEnvironment + SDK read."""
        if domain := self._config.get("domain"):
            os.environ["E2B_DOMAIN"] = domain
        if api_key := self._config.get("api_key"):
            os.environ["E2B_API_KEY"] = api_key
        elif api_key_file := self._config.get("api_key_file"):
            key = Path(api_key_file).expanduser().read_text().strip()
            if not key:
                raise ValueError(f"e2b api_key_file is empty: {api_key_file}")
            os.environ["E2B_API_KEY"] = key

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


def validate_api_key(api_key: str) -> None:
    """Validate an e2b API key by `e2b_` prefix + non-empty body only.

    The SDK normally requires hex keys; self-hosted clusters mint `[a-z0-9]`
    ones. Installed onto `e2b.api.validate_api_key` at import (bottom of file).
    """
    if not isinstance(api_key, str) or not api_key.startswith("e2b_") or len(api_key) <= 4:
        raise AuthenticationException('Invalid API key format: expected "e2b_" followed by a non-empty body.')


# Relax the SDK's hex-only key check before any Sandbox is constructed; self-hosted clusters use a broader key alphabet.
_e2b_api.validate_api_key = validate_api_key
