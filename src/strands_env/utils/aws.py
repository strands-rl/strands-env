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
import os

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from strands_env.utils.decorators import cache_by

logger = logging.getLogger(__name__)

type BotoClient = BaseClient
type BotoClientConfig = Config


def resolve_region_name(region_name: str | None = None, profile_name: str | None = None) -> str:
    """Resolve the AWS region name depending on the provided arguments.

    Args:
        region_name: AWS region name. If `None`, resolved in order of precedence:
            - `AWS_REGION` env var
            - `AWS_DEFAULT_REGION` env var
            - profile region from `~/.aws/config`
            - `us-east-1` fallback
        profile_name: Optional AWS profile name from `~/.aws/config`.
    """
    return (
        region_name
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or boto3.Session(profile_name=profile_name).region_name
        or "us-east-1"
    )


def get_session(
    region_name: str | None = None,
    profile_name: str | None = None,
    role_arn: str | None = None,
    role_session_name: str = "strands-env",
) -> boto3.Session:
    """Create a new boto3 session.

    Args:
        region_name: AWS region name. If `None`, resolved via `resolve_region_name`.
        profile_name: Optional AWS profile name from `~/.aws/config`.
        role_arn: Optional ARN of the IAM role to assume.
        role_session_name: Session name for assumed role (only used if role_arn provided).

    Notes:
        - Returns a **fresh** session every time — `boto3` sessions are not thread-safe,
          so they must not be shared across concurrent calls. Use `get_client` instead
          when you need a cached, thread-safe client.
        - If `role_arn` is provided, assumes the role using STS with auto-refreshing
          credentials via botocore's `RefreshableCredentials`.
    """
    region_name = resolve_region_name(region_name=region_name, profile_name=profile_name)
    if role_arn:
        return create_assumed_role_session(
            role_arn=role_arn, role_session_name=role_session_name, region_name=region_name
        )
    session = boto3.Session(region_name=region_name, profile_name=profile_name)
    logger.info("Created boto3 session: region_name=%s, profile_name=%s", session.region_name, session.profile_name)
    return session


def create_assumed_role_session(role_arn: str, role_session_name: str, region_name: str) -> boto3.Session:
    """Create a boto3 session with assumed role credentials."""
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session as get_botocore_session

    def refresh() -> dict:
        logger.info("Refreshing STS credentials for assumed role: %s", role_arn)
        sts = boto3.client("sts", region_name=region_name)
        creds = sts.assume_role(RoleArn=role_arn, RoleSessionName=role_session_name)["Credentials"]
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
    session = boto3.Session(botocore_session=botocore_session, region_name=region_name)
    logger.info("Created boto3 session with assumed role: role_arn=%s, region_name=%s", role_arn, session.region_name)
    return session


@cache_by("service_name", "region_name", "profile_name", "role_arn", "role_session_name")
def get_client(
    service_name: str,
    region_name: str | None = None,
    profile_name: str | None = None,
    role_arn: str | None = None,
    role_session_name: str = "strands-env",
    config: BotoClientConfig | None = None,
) -> BotoClient:
    """Get a cached boto3 client.

    Args:
        service_name: AWS service name (e.g. `"bedrock-agentcore"`, `"lambda"`, `"dynamodb"`).
        region_name: AWS region name. If `None`, resolved via `resolve_region_name`.
        profile_name: Optional AWS profile name from `~/.aws/config`.
        role_arn: Optional ARN of the IAM role to assume.
        role_session_name: Session name for assumed role (only used if role_arn provided).
        config: Optional `botocore.config.Config` for client-level settings
            (e.g. `max_pool_connections`, `connect_timeout`, `read_timeout`, `retries`).

    Notes:
        - Cached by `(service_name, region_name, profile_name, role_arn, role_session_name)`.
          `config` is excluded from the cache key (not hashable).
        - Each client gets its own dedicated boto3 Session, avoiding the thread-safety
          issues of sharing a Session across clients. The client itself is thread-safe.
        - If `role_arn` is provided, the underlying Session uses `RefreshableCredentials`
          so the client auto-refreshes when credentials expire.
    """
    region_name = resolve_region_name(region_name=region_name, profile_name=profile_name)
    if role_arn:
        session = create_assumed_role_session(
            role_arn=role_arn, role_session_name=role_session_name, region_name=region_name
        )
    else:
        session = boto3.Session(region_name=region_name, profile_name=profile_name)
    client = session.client(service_name, region_name=region_name, config=config)
    logger.info("Created cached boto3 client: service_name=%s, region_name=%s", service_name, client.meta.region_name)
    return client


def check_credentials(session: boto3.Session) -> bool:
    """Check whether a boto3 session has valid credentials."""
    try:
        session.client("sts").get_caller_identity()
        return True
    except Exception:
        return False
