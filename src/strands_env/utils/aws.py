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

"""Utilities for AWS boto3 session."""

from __future__ import annotations

import logging
from typing import TypeAlias

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from strands_env.utils.decorators import cache_by

logger = logging.getLogger(__name__)

BotoClient: TypeAlias = BaseClient
BotoClientConfig: TypeAlias = Config


def get_session(
    region: str = "us-west-2",
    profile_name: str | None = None,
    role_arn: str | None = None,
    session_name: str = "strands-env",
) -> boto3.Session:
    """Create a new boto3 session.

    Args:
        region: AWS region name.
        profile_name: Optional AWS profile name from ~/.aws/config.
        role_arn: Optional ARN of the IAM role to assume.
        session_name: Session name for assumed role (only used if role_arn provided).

    Notes:
        - Returns a **fresh** session every time — boto3 sessions are not thread-safe,
          so they must not be shared across concurrent calls. Use `get_client` instead
          when you need a cached, thread-safe client.
        - If `role_arn` is provided, assumes the role using STS with auto-refreshing
          credentials via botocore's `RefreshableCredentials`.
    """
    if role_arn:
        return create_assumed_role_session(role_arn, region, session_name)
    else:
        logger.info("Creating boto3 session: region=%s, profile=%s", region, profile_name)
        return boto3.Session(region_name=region, profile_name=profile_name)


def create_assumed_role_session(role_arn: str, region: str, session_name: str) -> boto3.Session:
    """Create a boto3 session with assumed role credentials."""
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session as get_botocore_session

    logger.info("Creating boto3 session with assumed role: role=%s, region=%s", role_arn, region)

    def refresh() -> dict:
        logger.info("Refreshing STS credentials for assumed role: %s", role_arn)
        sts = boto3.client("sts", region_name=region)
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName=session_name)["Credentials"]
        return {
            "access_key": creds["AccessKeyId"],
            "secret_key": creds["SecretAccessKey"],
            "token": creds["SessionToken"],
            "expiry_time": creds["Expiration"].isoformat(),
        }

    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=refresh(),
        refresh_using=refresh,
        method="sts-assume-role",
    )

    botocore_session = get_botocore_session()
    botocore_session._credentials = session_credentials
    return boto3.Session(botocore_session=botocore_session, region_name=region)


@cache_by("service_name", "region", "profile_name", "role_arn", "session_name")
def get_client(
    service_name: str,
    region: str = "us-east-1",
    profile_name: str | None = None,
    role_arn: str | None = None,
    session_name: str = "strands-env",
    config: BotoClientConfig | None = None,
) -> BotoClient:
    """Get a cached boto3 client.

    Args:
        service_name: AWS service name (e.g. `"bedrock-agentcore"`, `"lambda"`, `"dynamodb"`).
        region: AWS region name.
        profile_name: Optional AWS profile name from ~/.aws/config.
        role_arn: Optional ARN of the IAM role to assume.
        session_name: Session name for assumed role (only used if role_arn provided).
        config: Optional `botocore.config.Config` for client-level settings
            (e.g. `max_pool_connections`, `connect_timeout`, `read_timeout`, `retries`).

    Notes:
        - Cached by `(service_name, region, profile_name, role_arn, session_name)`.
          `config` is excluded from the cache key (not hashable).
        - Each client gets its own dedicated boto3 Session, avoiding the thread-safety
          issues of sharing a Session across clients. The client itself is thread-safe.
        - If `role_arn` is provided, the underlying Session uses `RefreshableCredentials`
          so the client auto-refreshes when credentials expire.
    """
    if role_arn:
        session = create_assumed_role_session(role_arn, region, session_name)
    else:
        session = boto3.Session(region_name=region, profile_name=profile_name)
    logger.info("Creating cached boto3 client: service=%s, region=%s", service_name, region)
    return session.client(service_name, region_name=region, config=config)


def check_credentials(session: boto3.Session) -> bool:
    """Check whether a boto3 session has valid credentials."""
    try:
        session.client("sts").get_caller_identity()
        return True
    except Exception:
        return False
