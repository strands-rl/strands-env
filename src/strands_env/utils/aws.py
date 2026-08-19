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
    """Resolve the AWS region name.

    In order: the `region_name` argument, `AWS_REGION`, `AWS_DEFAULT_REGION`, the profile's region
    from `~/.aws/config`, then `us-east-1`.
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
    """Create a new boto3 session, resolving the region via `resolve_region_name`.

    With `role_arn`, the role is assumed via STS with auto-refreshing credentials. Passing
    `profile_name` too makes it a two-hop chain — the role is assumed *from that profile's*
    credentials rather than the ambient ones, which is what a trust policy scoped to the profile's
    account requires.

    Notes:
        A **fresh** session every call. boto3 sessions are not thread-safe and must not be shared
        across concurrent calls; use `get_client` when you want a cached, thread-safe client.
    """
    region_name = resolve_region_name(region_name=region_name, profile_name=profile_name)
    if role_arn:
        return create_assumed_role_session(
            role_arn=role_arn,
            role_session_name=role_session_name,
            region_name=region_name,
            profile_name=profile_name,
        )
    session = boto3.Session(region_name=region_name, profile_name=profile_name)
    logger.info("Created boto3 session: region_name=%s, profile_name=%s", session.region_name, session.profile_name)
    return session


def create_assumed_role_session(
    role_arn: str, role_session_name: str, region_name: str, profile_name: str | None = None
) -> boto3.Session:
    """Create a boto3 session with assumed role credentials.

    `profile_name` selects the credentials the STS `AssumeRole` call is made with;
    `None` uses the ambient credential chain.
    """
    from botocore.credentials import RefreshableCredentials
    from botocore.session import get_session as get_botocore_session

    def refresh() -> dict:
        logger.info("Refreshing STS credentials for assumed role: %s (profile=%s)", role_arn, profile_name)
        sts = boto3.Session(region_name=region_name, profile_name=profile_name).client("sts", region_name=region_name)
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
    """Get a cached boto3 client, one dedicated session per client.

    Each client owns its session, so nothing shares the non-thread-safe `Session` while the clients
    themselves stay thread-safe. With `role_arn` the session uses `RefreshableCredentials`, so the
    client keeps working past credential expiry.

    Notes:
        `config` is excluded from the cache key because it isn't hashable — two calls that differ
        only in `config` return the first one's client.
    """
    region_name = resolve_region_name(region_name=region_name, profile_name=profile_name)
    if role_arn:
        session = create_assumed_role_session(
            role_arn=role_arn,
            role_session_name=role_session_name,
            region_name=region_name,
            profile_name=profile_name,
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
