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

"""Unit tests for the Harbor `e2b` backend adapter (`e2b.py`).

Covers the harbor-independent logic: template resolution, connection env-var
export, the permissive api-key validator, and the `PrebakedE2BEnvironment`
`__init__`/`start` overrides. Harbor's `E2BEnvironment` base and the e2b SDK are
not exercised against a real cluster — sandbox creation is mocked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The adapter subclasses harbor's E2BEnvironment, so the module can't import
# without harbor. Skip the whole file when harbor isn't installed (matches the
# convention in tests/integration/test_harbor.py).
pytest.importorskip("harbor", reason="harbor>=0.13.2 required for the e2b backend adapter")

from e2b.exceptions import AuthenticationException  # noqa: E402

from strands_env.environments.harbor import e2b  # noqa: E402
from strands_env.environments.harbor.e2b import PrebakedE2BEnvironment  # noqa: E402

#: `validate_api_key` is a module-level function; bind it for readable tests.
validate_api_key = e2b.validate_api_key


def _bare_env(*, task_id: str = "fix-git", config: dict | None = None) -> PrebakedE2BEnvironment:
    """A `PrebakedE2BEnvironment` with only `task_id`/`_config` set, bypassing __init__.

    `_resolve_template_id` and `_set_e2b_env_vars` are private instance methods
    that read only those two attributes, so this drives them in isolation without
    touching the e2b SDK or harbor's base `__init__`.
    """
    env = PrebakedE2BEnvironment.__new__(PrebakedE2BEnvironment)
    env.task_id = task_id
    env._config = config or {}
    return env


def _resolve_template_id(task_id: str, config: dict) -> str:
    """Drive `_resolve_template_id` on a bare instance."""
    return _bare_env(task_id=task_id, config=config)._resolve_template_id()


def _set_e2b_env_vars(config: dict) -> None:
    """Drive `_set_e2b_env_vars` on a bare instance."""
    _bare_env(config=config)._set_e2b_env_vars()


# ---------------------------------------------------------------------------
# validate_api_key
# ---------------------------------------------------------------------------


class TestPermissiveValidateApiKey:
    """The relaxed api-key validator monkey-patched onto the e2b SDK."""

    def test_accepts_hex_key(self):
        """A standard hex key passes (no-op for upstream keys)."""
        validate_api_key("e2b_0123456789abcdef")  # no raise

    def test_accepts_non_hex_alnum_key(self):
        """A self-hosted `[a-z0-9]` key passes (the reason the patch exists)."""
        validate_api_key("e2b_3fw7jq32e71yfqu0")  # no raise

    def test_rejects_missing_prefix(self):
        """A key without the `e2b_` prefix is rejected."""
        with pytest.raises(AuthenticationException):
            validate_api_key("sk_0123456789")

    def test_rejects_empty_body(self):
        """`e2b_` with no body (len <= 4) is rejected."""
        with pytest.raises(AuthenticationException):
            validate_api_key("e2b_")

    def test_rejects_non_string(self):
        """A non-string key is rejected, not crashed on."""
        with pytest.raises(AuthenticationException):
            validate_api_key(None)  # type: ignore[arg-type]

    def test_patch_is_installed_on_e2b_sdk(self):
        """Importing the module installs the validator on e2b.api at module load."""
        assert _e2b_api_validator() is validate_api_key


def _e2b_api_validator():
    import e2b.api as api

    return api.validate_api_key


# ---------------------------------------------------------------------------
# _resolve_template_id
# ---------------------------------------------------------------------------


class TestResolveTemplateId:
    """Resolving a task name to its template id (`_resolve_template_id`)."""

    @pytest.fixture
    def templates(self, tmp_path: Path) -> Path:
        p = tmp_path / "templates.json"
        p.write_text(json.dumps({"fix-git": "tmpl-fixgit", "make-doc": "tmpl-makedoc"}))
        return p

    def test_direct_name_hit(self, templates: Path):
        """A bare task name resolves directly."""
        assert _resolve_template_id("fix-git", {"templates_json": str(templates)}) == "tmpl-fixgit"

    def test_benchmark_prefixed_name_falls_back_to_bare(self, templates: Path):
        """Harbor's `<benchmark>/<task>` form falls back to the bare dir name."""
        assert _resolve_template_id("terminal-bench/fix-git", {"templates_json": str(templates)}) == "tmpl-fixgit"

    def test_reads_env_var_when_config_has_no_path(self, templates: Path, monkeypatch):
        """With no `templates_json` in config, E2B_TEMPLATES_PATH is used."""
        monkeypatch.setenv("E2B_TEMPLATES_PATH", str(templates))
        assert _resolve_template_id("make-doc", {}) == "tmpl-makedoc"

    def test_missing_env_var_and_path_raises_runtimeerror(self, monkeypatch):
        """No `templates_json` + unset env var raises an actionable RuntimeError."""
        monkeypatch.delenv("E2B_TEMPLATES_PATH", raising=False)
        with pytest.raises(RuntimeError, match="E2B_TEMPLATES_PATH not set"):
            _resolve_template_id("fix-git", {})

    def test_unknown_task_raises_keyerror(self, templates: Path):
        """A task absent from the mapping raises KeyError listing what was tried."""
        with pytest.raises(KeyError, match="has no matching template"):
            _resolve_template_id("never-baked", {"templates_json": str(templates)})

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        """A nonexistent templates path surfaces FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _resolve_template_id("fix-git", {"templates_json": str(tmp_path / "nope.json")})

    def test_config_path_overrides_env_var(self, templates: Path, tmp_path: Path, monkeypatch):
        """A `templates_json` in config wins over E2B_TEMPLATES_PATH."""
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"fix-git": "from-env-var"}))
        monkeypatch.setenv("E2B_TEMPLATES_PATH", str(other))
        assert _resolve_template_id("fix-git", {"templates_json": str(templates)}) == "tmpl-fixgit"


# ---------------------------------------------------------------------------
# PrebakedE2BEnvironment.__init__
# ---------------------------------------------------------------------------


class TestPrebakedE2BEnvironmentInit:
    """__init__: resolve the pinned template id + export connection env vars."""

    def test_explicit_template_id_skips_resolution(self):
        """An explicit `template_id` is used directly and the connection env is exported."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(e2b.E2BEnvironment, "__init__", return_value=None) as base_init,
        ):
            env = PrebakedE2BEnvironment(
                task_id="fix-git",
                prebaked_e2b_config={"template_id": "tmpl-explicit", "domain": "e2b.acme.dev"},
            )
            assert env._template_id == "tmpl-explicit"
            assert os.environ["E2B_DOMAIN"] == "e2b.acme.dev"
        base_init.assert_called_once()

    def test_resolves_template_from_map(self, tmp_path: Path):
        """With no `template_id`, the id is resolved from `templates_json` by task name."""
        templates = tmp_path / "templates.json"
        templates.write_text(json.dumps({"fix-git": "tmpl-resolved"}))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(e2b.E2BEnvironment, "__init__", return_value=None),
        ):
            env = PrebakedE2BEnvironment(
                task_id="fix-git",
                prebaked_e2b_config={"templates_json": str(templates)},
            )
        assert env._template_id == "tmpl-resolved"


# ---------------------------------------------------------------------------
# PrebakedE2BEnvironment.start
# ---------------------------------------------------------------------------


def _make_env() -> PrebakedE2BEnvironment:
    """Build a PrebakedE2BEnvironment without running harbor's heavy __init__.

    We bypass __init__ (which would touch the e2b SDK / harbor internals) and
    set only the attributes the methods under test read, so the override logic
    is exercised in isolation.
    """
    env = PrebakedE2BEnvironment.__new__(PrebakedE2BEnvironment)
    env._template_id = "tmpl-pinned"
    env._sandbox = MagicMock()
    env.logger = MagicMock()
    return env


class TestPrebakedE2BEnvironmentStart:
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


# ---------------------------------------------------------------------------
# _set_e2b_env_vars
# ---------------------------------------------------------------------------


class TestSetE2BEnvVars:
    """Writing the e2b connection (domain/api_key) into the env vars the SDK reads."""

    def test_sets_domain_and_api_key(self):
        """`domain`/`api_key` land in E2B_DOMAIN/E2B_API_KEY."""
        with patch.dict(os.environ, {}, clear=True):
            _set_e2b_env_vars({"domain": "e2b.acme.dev", "api_key": "e2b_abc"})
            assert os.environ["E2B_DOMAIN"] == "e2b.acme.dev"
            assert os.environ["E2B_API_KEY"] == "e2b_abc"

    def test_api_key_file_is_read_and_stripped(self, tmp_path: Path):
        """`api_key_file` is read and stripped into E2B_API_KEY."""
        key_file = tmp_path / "key"
        key_file.write_text("  e2b_fromfile\n")
        with patch.dict(os.environ, {}, clear=True):
            _set_e2b_env_vars({"api_key_file": str(key_file)})
            assert os.environ["E2B_API_KEY"] == "e2b_fromfile"

    def test_api_key_wins_over_api_key_file(self, tmp_path: Path):
        """An explicit `api_key` takes precedence; the file is not consulted."""
        key_file = tmp_path / "key"
        key_file.write_text("e2b_fromfile")
        with patch.dict(os.environ, {}, clear=True):
            _set_e2b_env_vars({"api_key": "e2b_inline", "api_key_file": str(key_file)})
            assert os.environ["E2B_API_KEY"] == "e2b_inline"

    def test_empty_api_key_file_raises(self, tmp_path: Path):
        """A blank `api_key_file` is a hard error, not a silent empty key."""
        key_file = tmp_path / "key"
        key_file.write_text("   \n")
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="api_key_file is empty"),
        ):
            _set_e2b_env_vars({"api_key_file": str(key_file)})

    def test_empty_config_is_noop(self):
        """An empty config writes nothing."""
        with patch.dict(os.environ, {}, clear=True):
            _set_e2b_env_vars({})
            assert "E2B_DOMAIN" not in os.environ
            assert "E2B_API_KEY" not in os.environ
