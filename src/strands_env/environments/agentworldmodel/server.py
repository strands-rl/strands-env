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

"""AgentWorldModel server script generation and lifecycle utilities."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from awm.tools import get_random_available_port, normalize_scenario_name

logger = logging.getLogger(__name__)

_MCP_INJECT = """\
    from fastapi_mcp import FastApiMCP
    mcp = FastApiMCP(app)
    mcp.mount_http()"""
SERVER_STARTUP_TIMEOUT = 30


def write_server_script(
    script_path: Path,
    port: int,
    scenario: str,
    envs_path: Path,
    work_db_path: Path,
) -> None:
    """Generate a runnable FastAPI server script from gen_envs.jsonl.

    Same transformation as `awm.core.server.run_server()` but writes a
    self-contained script — no `os.system()`, no intermediate processes.
    """
    normalized = normalize_scenario_name(scenario)
    with open(envs_path) as f:
        for line in f:
            entry = json.loads(line)
            if normalize_scenario_name(entry["scenario"]) == normalized:
                break
        else:
            raise ValueError(f"Scenario {normalized} not found in {envs_path}")

    new_lines = ["import warnings", 'warnings.filterwarnings("ignore", category=DeprecationWarning)']
    for src_line in entry["full_code"].split("\n"):
        if "create_engine(" in src_line:
            left = src_line.split("create_engine(")[0]
            src_line = f"{left}create_engine('sqlite:///{work_db_path}', connect_args={{'check_same_thread': False}})"
        if "uvicorn.run(app" in src_line:
            new_lines.append(_MCP_INJECT)
            src_line = f"    uvicorn.run(app, host='127.0.0.1', port={port})"
        new_lines.append(src_line)

    script_path.write_text("\n".join(new_lines))


def _wait_for_server_sync(port: int, scenario: str) -> None:
    """Block until the server accepts TCP connections (sync, runs in thread)."""
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    raise TimeoutError(f"AgentWorldModel server for {scenario} did not start within {SERVER_STARTUP_TIMEOUT}s")


async def wait_for_server(port: int, scenario: str) -> None:
    """Async wrapper — runs sync poll in a thread to keep fds off the event loop."""
    await asyncio.to_thread(_wait_for_server_sync, port, scenario)


async def start_server(
    scenario: str,
    envs_path: Path,
    work_db_path: Path,
    temp_dir: Path,
) -> tuple[subprocess.Popen, int]:
    """Generate server script, start subprocess, wait for TCP readiness.

    Returns:
        A tuple of (server process, port).
    """
    port = get_random_available_port()
    script = temp_dir / "server.py"
    await asyncio.to_thread(write_server_script, script, port, scenario, envs_path, work_db_path)

    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    await wait_for_server(port, scenario)
    logger.info("AgentWorldModel server pid=%d for %s on port %d", proc.pid, scenario, port)
    return proc, port


async def kill_server(proc: subprocess.Popen | None, timeout: float = 5) -> None:
    """Graceful shutdown: SIGTERM first, then SIGKILL if needed."""
    if proc is None or proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        await asyncio.to_thread(proc.wait, timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(proc.wait, timeout=2)
