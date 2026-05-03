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

"""harbor-aws session preempt for long-running `/exec` calls.

Installs a keepalive-enabled aiohttp `ClientSession` with a permissive timeout into
`AWSEnvironment._shared_aiohttp_session` so that long-running `/exec` calls aren't
truncated by aiohttp's 300s default and aren't dropped by AWS NLB's ~350s idle timeout.
"""

from __future__ import annotations

import asyncio
import socket

import aiohttp
from harbor_aws.adapter import AWSEnvironment

# Mirrors `AWSEnvironment.__init__`'s `pod_timeout_sec` default. If callers
# override that via `eks_backend_config`, update this to match.
_HARBOR_POD_TIMEOUT_SEC = 14400


# TCP keepalive intervals tuned to fire well before AWS NLB's ~350s idle
# timeout, so the LB can't silently drop in-flight `/exec` connections.
# `TCP_KEEPIDLE`/`TCP_KEEPINTVL`/`TCP_KEEPCNT` are Linux-only; on other
# platforms (e.g. macOS dev boxes) we degrade to plain `SO_KEEPALIVE` whose
# defaults are too long to help with NLB but won't break the connection.
def _make_keepalive_socket(addr_info: tuple) -> socket.socket:  # type: ignore[type-arg]
    family, type_, proto, *_ = addr_info
    s = socket.socket(family, type_, proto)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 30), ("TCP_KEEPCNT", 4)):
        const = getattr(socket, name, None)
        if const is not None:
            s.setsockopt(socket.IPPROTO_TCP, const, value)
    return s


async def ensure_harbor_aws_session() -> None:
    """Install a keepalive-enabled aiohttp session into harbor-aws.

    harbor-aws 0.4.0 lazily creates its process-wide shared session with
    aiohttp's default `ClientTimeout(total=300)` and no TCP keepalive, which
    caps tool calls at 5 minutes and leaves `/exec` calls vulnerable to AWS
    NLB silently dropping idle connections at its ~350s timeout. We pre-
    populate `AWSEnvironment._shared_aiohttp_session` with a session that has
    TCP keepalive and a 4-hour `total` matching harbor's `pod_timeout_sec`.

    Idempotent. Must be awaited before the first `AWSEnvironment.start()`.
    """
    if AWSEnvironment._shared_aiohttp_session is not None:
        return
    if AWSEnvironment._shared_aiohttp_lock is None:
        AWSEnvironment._shared_aiohttp_lock = asyncio.Lock()
    async with AWSEnvironment._shared_aiohttp_lock:
        if AWSEnvironment._shared_aiohttp_session is not None:  # type: ignore[unreachable]
            return  # type: ignore[unreachable]
        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=0,
            socket_factory=_make_keepalive_socket,
        )
        AWSEnvironment._shared_aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=_HARBOR_POD_TIMEOUT_SEC,
                sock_connect=30,
            ),
        )
