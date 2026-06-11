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

"""e2b-on-AWS template adapter for the Harbor `e2b` backend.

Some self-hosted e2b deployments don't implement Harbor's auto-build template
route. This adapter subclasses Harbor's `E2BEnvironment` to skip that path and
boot from a `template_id` resolved from a `templates.json` mapping (or the
`E2B_TEMPLATES_PATH` env var).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import e2b.api as _e2b_api
import httpx
from e2b.exceptions import AuthenticationException
from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.paths import EnvironmentPaths
from typing_extensions import override

if TYPE_CHECKING:
    from harbor.models.task.config import EnvironmentConfig as HarborEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths

logger = logging.getLogger(__name__)


def _permissive_validate_api_key(api_key: str) -> None:
    if not isinstance(api_key, str) or not api_key.startswith("e2b_") or len(api_key) <= 4:
        raise AuthenticationException('Invalid API key format: expected "e2b_" followed by a non-empty body.')


# The upstream e2b SDK validates API keys against a hex-only regex. Some
# self-hosted deployments mint keys from a broader `[a-z0-9]` alphabet, which
# the SDK would reject client-side. Relax the validator to only require the
# `e2b_` prefix + non-empty body. No-op for standard hex keys. Applied at import
# time, before harbor's E2BEnvironment constructs any Sandbox.
_e2b_api.validate_api_key = _permissive_validate_api_key

_RETRYABLE_HTTP2_ERRORS: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
)

T = TypeVar("T")


async def _retry_on_http2_transient(
    op: Callable[[], Awaitable[T]],
    *,
    op_name: str,
    max_attempts: int = 3,
    initial_backoff_s: float = 0.05,
) -> T:
    """Run *op* with bounded exponential backoff on HTTP/2 transients.

    Retries only on the classes in `_RETRYABLE_HTTP2_ERRORS`; any other
    exception is re-raised immediately so test failures, missing files, etc.
    are not silently retried. Backoff is exponential with jitter starting at
    50ms.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await op()
        except _RETRYABLE_HTTP2_ERRORS as e:
            last_exc = e
            if attempt == max_attempts:
                logger.warning(
                    "%s: giving up after %d attempts on %s: %s",
                    op_name,
                    attempt,
                    type(e).__name__,
                    e,
                )
                raise
            sleep_s = initial_backoff_s * (2 ** (attempt - 1)) * (1 + random.random())
            logger.info(
                "%s: retry %d/%d after %s: %s (sleep %.2fs)",
                op_name,
                attempt,
                max_attempts,
                type(e).__name__,
                e,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)
    # Unreachable — final attempt either returned or re-raised. Satisfy type checker.
    assert last_exc is not None
    raise last_exc


def _load_template_map(path: str | Path) -> dict[str, str]:
    """Load a {task_name: template_id} mapping from a JSON file.

    The file maps each Harbor task name to a template id baked against the
    self-hosted e2b cluster.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Templates file not found at {p}. Bake the task templates against the e2b cluster first.",
        )
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object mapping task_name -> template_id.")
    return data


def resolve_template_id(task_name: str, template_map_path: str | Path | None = None) -> str:
    """Resolve a Harbor task name to its e2b template id.

    Args:
        task_name: The task's name (e.g. `"fix-git"`). Matches the directory
            name in the dataset and the `Task.name` field after Harbor's
            mapper has run.
        template_map_path: Path to `templates.json`. If None, reads from
            `E2B_TEMPLATES_PATH`.

    Raises:
        FileNotFoundError: If the templates file does not exist.
        KeyError: If the task name has no entry. A template must be baked for
            new tasks before they can be evaluated.
    """
    if template_map_path is None:
        template_map_path = os.environ.get("E2B_TEMPLATES_PATH")
    if template_map_path is None:
        raise RuntimeError(
            "E2B_TEMPLATES_PATH not set and no template_map_path provided. "
            "Set it to a templates.json mapping task_name -> template_id.",
        )
    mapping = _load_template_map(template_map_path)
    # Tolerate both Harbor's `<benchmark>/<task>` form (e.g. `terminal-bench/fix-git`)
    # and the bare directory name (`fix-git`). The mapping uses the bare directory
    # name; the evaluator passes Harbor's full Task.name.
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


class E2bAWSEnvironment(E2BEnvironment):
    """E2BEnvironment that skips Harbor's auto-build path.

    Construction is identical to `E2BEnvironment` except a `template_id` kwarg
    pins the template the sandbox should boot from.
    """

    # Harbor's reward.py checks `docker_env.is_mounted` to decide whether
    # results need to be downloaded out of the environment. Mounted
    # filesystems (Docker on Linux + bind mounts) skip the download because
    # the host already sees the files. e2b sandboxes are remote — files only
    # exist inside the microVM until downloaded, so this must be False.
    is_mounted = False

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: HarborEnvironmentConfig,
        *args: Any,
        template_id: str,
        **kwargs: Any,
    ) -> None:
        # Forward the five required fields positionally so any extra *args
        # unpack cleanly after them. Passing them by keyword *before* `*args`
        # (B026) would raise "multiple values for argument" the moment a
        # positional extra is supplied, since `*args` binds to the same leading
        # parameters.
        super().__init__(
            environment_dir,
            environment_name,
            session_id,
            trial_paths,
            task_env_config,
            *args,
            **kwargs,
        )
        self._template_id = template_id

    @override
    async def start(self, force_build: bool) -> None:
        # Skip _does_template_exist + _create_template. Use the resolved id.
        self._template_name = self._template_id
        if force_build:
            self.logger.warning(
                "force_build=True requested but E2bAWSEnvironment ignores it. "
                "Re-bake the template against the e2b cluster if needed.",
            )
        await self._create_sandbox()
        if not self._sandbox:
            raise RuntimeError("Sandbox not found but was just created.")
        # Harbor convention: tasks read/write under /logs/{agent,verifier,artifacts}
        # and /tests. The base E2BEnvironment.start() relies on Harbor's auto-build
        # path to bake these into the image; this adapter skips that path so we
        # must create them ourselves. Without these, reward.py's `tee /logs/
        # verifier/test-stdout.txt` and `download_dir(/logs/verifier)` fail with
        # FileNotFoundException for any task image that didn't pre-create the dirs.
        # User-declared mount targets are unioned in so explicit volumes still work.
        harbor_dirs = [
            EnvironmentPaths.logs_dir,
            EnvironmentPaths.agent_dir,
            EnvironmentPaths.verifier_dir,
            EnvironmentPaths.artifacts_dir,
            EnvironmentPaths.tests_dir,
        ]
        mount_targets = self._mount_targets(writable_only=True)
        await self.ensure_dirs([*harbor_dirs, *mount_targets])
        # Preserve base E2BEnvironment.start() semantics: prebuilt-image tasks
        # that ship supplemental files under environment/ (no Dockerfile /
        # compose) rely on this to copy them into the sandbox workdir. The
        # method self-guards (no-ops when should_upload_environment_dir is
        # False), so it's safe for tasks that don't need it.
        await self._upload_environment_dir_after_start()

    @override
    async def _create_template(self) -> None:  # type: ignore[override]
        # Surface a precise error if Harbor ever falls into this path (it
        # shouldn't, since we override start; but defensive).
        raise RuntimeError(
            "E2bAWSEnvironment does not auto-build templates. "
            "Templates must be baked against the e2b cluster before eval.",
        )

    # These four overrides wrap harbor's E2BEnvironment file-transfer calls in
    # the same bounded-retry policy `Sandbox.create` already uses internally,
    # so a transient HTTP/2 drop on a verifier round-trip (HarborReward.compute
    # → upload_dir + download_dir) doesn't drop a whole task from the pass@k
    # denominator. These ops are idempotent — re-running them re-transfers the
    # same bytes — so retry is always safe here.
    #
    # `exec` is deliberately NOT wrapped: a retry after a *post-dispatch*
    # transient (the command already started server-side, but the client failed
    # reading the response) would silently run the command twice. That's
    # harmless for the idempotent verifier `test.sh` but unsafe for arbitrary
    # agent commands, and `exec` can't tell the two callers apart. Retrying the
    # known-idempotent verifier exec is tracked separately (a future
    # `exec_idempotent` entry point the verifier opts into). Until then `exec`
    # inherits the base, un-retried behavior.

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await _retry_on_http2_transient(
            lambda: super(E2bAWSEnvironment, self).upload_file(source_path, target_path),
            op_name=f"upload_file({target_path})",
        )

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await _retry_on_http2_transient(
            lambda: super(E2bAWSEnvironment, self).upload_dir(source_dir, target_dir),
            op_name=f"upload_dir({target_dir})",
        )

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await _retry_on_http2_transient(
            lambda: super(E2bAWSEnvironment, self).download_file(source_path, target_path),
            op_name=f"download_file({source_path})",
        )

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await _retry_on_http2_transient(
            lambda: super(E2bAWSEnvironment, self).download_dir(source_dir, target_dir),
            op_name=f"download_dir({source_dir})",
        )
