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

"""Unit tests for the Harbor `e2b` backend adapter (`_e2b_aws.py`).

Covers the harbor-independent logic: template resolution, the templates.json
loader, the HTTP/2-transient retry helper, the permissive api-key validator,
and the `E2bAWSEnvironment` overrides (build-skip, Harbor dir creation, the
retry-wrapped file/exec methods). Harbor's `E2BEnvironment` base and the e2b
SDK are not exercised against a real cluster — sandbox creation is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# The adapter subclasses harbor's E2BEnvironment, so the module can't import
# without harbor. Skip the whole file when harbor isn't installed (matches the
# convention in tests/integration/test_harbor.py).
pytest.importorskip("harbor", reason="harbor>=0.1.43 required for the e2b backend adapter")

from e2b.exceptions import AuthenticationException  # noqa: E402

from strands_env.environments.harbor import _e2b_aws  # noqa: E402
from strands_env.environments.harbor._e2b_aws import (  # noqa: E402
    E2bAWSEnvironment,
    _load_template_map,
    _permissive_validate_api_key,
    _retry_on_http2_transient,
    resolve_template_id,
)

# ---------------------------------------------------------------------------
# _permissive_validate_api_key
# ---------------------------------------------------------------------------


class TestPermissiveValidateApiKey:
    """The relaxed api-key validator monkey-patched onto the e2b SDK."""

    def test_accepts_hex_key(self):
        """A standard hex key passes (no-op for upstream keys)."""
        _permissive_validate_api_key("e2b_0123456789abcdef")  # no raise

    def test_accepts_non_hex_alnum_key(self):
        """A self-hosted `[a-z0-9]` key passes (the reason the patch exists)."""
        _permissive_validate_api_key("e2b_3fw7jq32e71yfqu0")  # no raise

    def test_rejects_missing_prefix(self):
        """A key without the `e2b_` prefix is rejected."""
        with pytest.raises(AuthenticationException):
            _permissive_validate_api_key("sk_0123456789")

    def test_rejects_empty_body(self):
        """`e2b_` with no body (len <= 4) is rejected."""
        with pytest.raises(AuthenticationException):
            _permissive_validate_api_key("e2b_")

    def test_rejects_non_string(self):
        """A non-string key is rejected, not crashed on."""
        with pytest.raises(AuthenticationException):
            _permissive_validate_api_key(None)  # type: ignore[arg-type]

    def test_patch_is_installed_on_e2b_sdk(self):
        """Importing the module installs the validator on e2b.api at module load."""
        assert _e2b_api_validator() is _permissive_validate_api_key


def _e2b_api_validator():
    import e2b.api as api

    return api.validate_api_key


# ---------------------------------------------------------------------------
# _load_template_map
# ---------------------------------------------------------------------------


class TestLoadTemplateMap:
    """Loading + validating a templates.json file."""

    def test_loads_valid_mapping(self, tmp_path: Path):
        """A JSON object is returned as a dict."""
        p = tmp_path / "templates.json"
        p.write_text(json.dumps({"fix-git": "tmpl-abc", "make-doc": "tmpl-def"}))
        assert _load_template_map(p) == {"fix-git": "tmpl-abc", "make-doc": "tmpl-def"}

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        """A nonexistent path raises FileNotFoundError with guidance."""
        with pytest.raises(FileNotFoundError, match="Templates file not found"):
            _load_template_map(tmp_path / "nope.json")

    def test_non_dict_json_raises_valueerror(self, tmp_path: Path):
        """A JSON array (or any non-object) raises a clear ValueError."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(["fix-git", "make-doc"]))
        with pytest.raises(ValueError, match="must contain a JSON object"):
            _load_template_map(p)

    def test_accepts_str_path(self, tmp_path: Path):
        """A str path (not just Path) works."""
        p = tmp_path / "templates.json"
        p.write_text(json.dumps({"t": "id"}))
        assert _load_template_map(str(p)) == {"t": "id"}


# ---------------------------------------------------------------------------
# resolve_template_id
# ---------------------------------------------------------------------------


class TestResolveTemplateId:
    """Resolving a task name to its template id."""

    @pytest.fixture
    def templates(self, tmp_path: Path) -> Path:
        p = tmp_path / "templates.json"
        p.write_text(json.dumps({"fix-git": "tmpl-fixgit", "make-doc": "tmpl-makedoc"}))
        return p

    def test_direct_name_hit(self, templates: Path):
        """A bare task name resolves directly."""
        assert resolve_template_id("fix-git", templates) == "tmpl-fixgit"

    def test_benchmark_prefixed_name_falls_back_to_bare(self, templates: Path):
        """Harbor's `<benchmark>/<task>` form falls back to the bare dir name."""
        assert resolve_template_id("terminal-bench/fix-git", templates) == "tmpl-fixgit"

    def test_reads_env_var_when_path_is_none(self, templates: Path, monkeypatch):
        """With no explicit path, E2B_TEMPLATES_PATH is used."""
        monkeypatch.setenv("E2B_TEMPLATES_PATH", str(templates))
        assert resolve_template_id("make-doc") == "tmpl-makedoc"

    def test_missing_env_var_and_path_raises_runtimeerror(self, monkeypatch):
        """No path + unset env var raises an actionable RuntimeError."""
        monkeypatch.delenv("E2B_TEMPLATES_PATH", raising=False)
        with pytest.raises(RuntimeError, match="E2B_TEMPLATES_PATH not set"):
            resolve_template_id("fix-git")

    def test_unknown_task_raises_keyerror(self, templates: Path):
        """A task absent from the mapping raises KeyError listing what was tried."""
        with pytest.raises(KeyError, match="has no template"):
            resolve_template_id("never-baked", templates)

    def test_explicit_path_overrides_env_var(self, templates: Path, tmp_path: Path, monkeypatch):
        """An explicit template_map_path wins over E2B_TEMPLATES_PATH."""
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"fix-git": "from-env-var"}))
        monkeypatch.setenv("E2B_TEMPLATES_PATH", str(other))
        # explicit `templates` maps fix-git -> tmpl-fixgit, not from-env-var
        assert resolve_template_id("fix-git", templates) == "tmpl-fixgit"


# ---------------------------------------------------------------------------
# _retry_on_http2_transient
# ---------------------------------------------------------------------------


class TestRetryOnHttp2Transient:
    """The bounded-retry helper for HTTP/2 transients."""

    async def test_returns_on_first_success(self):
        """A call that succeeds immediately runs exactly once."""
        op = AsyncMock(return_value="ok")
        result = await _retry_on_http2_transient(op, op_name="x", initial_backoff_s=0.0)
        assert result == "ok"
        assert op.call_count == 1

    async def test_retries_then_succeeds(self):
        """A retryable transient is retried until the call succeeds."""
        op = AsyncMock(side_effect=[httpx.ReadError("boom"), "ok"])
        result = await _retry_on_http2_transient(op, op_name="x", initial_backoff_s=0.0)
        assert result == "ok"
        assert op.call_count == 2

    async def test_exhausts_attempts_and_reraises(self):
        """All attempts failing re-raises the last transient after max_attempts."""
        op = AsyncMock(side_effect=httpx.ConnectError("always"))
        with pytest.raises(httpx.ConnectError):
            await _retry_on_http2_transient(op, op_name="x", max_attempts=3, initial_backoff_s=0.0)
        assert op.call_count == 3

    async def test_non_retryable_raises_immediately(self):
        """A non-transient error propagates on the first attempt (no retry)."""
        op = AsyncMock(side_effect=ValueError("nope"))
        with pytest.raises(ValueError):
            await _retry_on_http2_transient(op, op_name="x", initial_backoff_s=0.0)
        assert op.call_count == 1


# ---------------------------------------------------------------------------
# E2bAWSEnvironment
# ---------------------------------------------------------------------------


def _make_env() -> E2bAWSEnvironment:
    """Build an E2bAWSEnvironment without running harbor's heavy __init__.

    We bypass __init__ (which would touch the e2b SDK / harbor internals) and
    set only the attributes the methods under test read, so the override logic
    is exercised in isolation.
    """
    env = E2bAWSEnvironment.__new__(E2bAWSEnvironment)
    env._template_id = "tmpl-pinned"
    env._sandbox = MagicMock()
    env.logger = MagicMock()
    return env


class TestE2bAWSEnvironmentConstruction:
    """Constructor + class-level attributes."""

    def test_init_stores_template_id(self):
        """The `template_id` kwarg is recorded as `_template_id`."""
        with patch.object(_e2b_aws.E2BEnvironment, "__init__", return_value=None) as base_init:
            env = E2bAWSEnvironment(
                Path("/env"),
                "env-name",
                "sess-1",
                MagicMock(),  # trial_paths
                MagicMock(),  # task_env_config
                template_id="tmpl-xyz",
            )
        assert env._template_id == "tmpl-xyz"
        base_init.assert_called_once()

    def test_is_mounted_is_false(self):
        """e2b sandboxes are remote, so is_mounted must be False (drives result download)."""
        assert E2bAWSEnvironment.is_mounted is False


class TestE2bAWSEnvironmentStart:
    """The `start()` override: skip build, create dirs, upload env dir."""

    async def test_start_pins_template_and_creates_dirs(self):
        """start() sets _template_name to the pinned id and ensure_dirs the Harbor paths."""
        env = _make_env()
        env._create_sandbox = AsyncMock()
        env._mount_targets = MagicMock(return_value=["/work"])
        env.ensure_dirs = AsyncMock()
        env._upload_environment_dir_after_start = AsyncMock()

        await env.start(force_build=False)

        assert env._template_name == "tmpl-pinned"
        env._create_sandbox.assert_awaited_once()
        env.ensure_dirs.assert_awaited_once()
        # ensure_dirs gets the 5 standard Harbor dirs + the mount targets.
        passed = env.ensure_dirs.call_args.args[0]
        assert "/work" in passed
        assert len(passed) >= 6
        env._upload_environment_dir_after_start.assert_awaited_once()

    async def test_start_warns_but_ignores_force_build(self):
        """force_build=True is ignored (templates are static) but logged."""
        env = _make_env()
        env._create_sandbox = AsyncMock()
        env._mount_targets = MagicMock(return_value=[])
        env.ensure_dirs = AsyncMock()
        env._upload_environment_dir_after_start = AsyncMock()

        await env.start(force_build=True)

        env.logger.warning.assert_called_once()
        assert env._template_name == "tmpl-pinned"

    async def test_start_raises_when_sandbox_not_created(self):
        """If _create_sandbox leaves no sandbox, start() fails loudly."""
        env = _make_env()
        env._sandbox = None
        env._create_sandbox = AsyncMock()
        env._mount_targets = MagicMock(return_value=[])
        env.ensure_dirs = AsyncMock()

        with pytest.raises(RuntimeError, match="Sandbox not found"):
            await env.start(force_build=False)

    async def test_create_template_raises(self):
        """The auto-build path is disabled and raises a precise error."""
        env = _make_env()
        with pytest.raises(RuntimeError, match="does not auto-build templates"):
            await env._create_template()


class TestE2bAWSEnvironmentRetryWrappers:
    """The idempotent file-transfer overrides delegate to the base through the retry helper."""

    async def test_upload_file_delegates_to_base(self):
        """upload_file() routes through the base implementation."""
        env = _make_env()
        with patch.object(_e2b_aws.E2BEnvironment, "upload_file", new=AsyncMock()) as base_upload:
            await env.upload_file("/src", "/dst")
        base_upload.assert_awaited_once()

    async def test_upload_file_retries_transient_then_succeeds(self):
        """A transient on the first attempt is retried via the helper (idempotent op)."""
        env = _make_env()
        base_upload = AsyncMock(side_effect=[httpx.RemoteProtocolError("drop"), None])
        with patch.object(_e2b_aws.E2BEnvironment, "upload_file", new=base_upload):
            await env.upload_file("/src", "/dst")
        assert base_upload.await_count == 2

    async def test_download_dir_delegates_to_base(self):
        """download_dir() routes through the base implementation."""
        env = _make_env()
        with patch.object(_e2b_aws.E2BEnvironment, "download_dir", new=AsyncMock()) as base_download:
            await env.download_dir("/remote", "/local")
        base_download.assert_awaited_once()

    def test_exec_is_not_overridden(self):
        """`exec` must NOT be wrapped in retry: a post-dispatch retry could double-run a
        non-idempotent agent command. The adapter inherits the base, un-retried `exec`.
        (Retrying the known-idempotent verifier exec is tracked as separate work.)"""
        assert "exec" not in E2bAWSEnvironment.__dict__
