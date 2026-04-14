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

"""Tests for utils/aws.py and utils/decorators.py."""

import threading
import time
from unittest.mock import MagicMock, patch

import boto3
import pytest

from strands_env.utils.aws import check_credentials, get_client, get_session
from strands_env.utils.decorators import cache_by, requires_env, with_timeout

# ===========================================================================
# AWS — get_session
# ===========================================================================


class TestGetSession:
    def test_returns_session(self):
        session = get_session(region="us-west-2")
        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    def test_fresh_session_each_call(self):
        session1 = get_session(region="us-east-1")
        session2 = get_session(region="us-east-1")
        assert session1 is not session2

    @patch("strands_env.utils.aws.boto3.Session")
    def test_passes_profile(self, mock_session_cls):
        mock_session_cls.return_value = MagicMock()
        get_session(region="us-east-1", profile_name="test-profile")
        mock_session_cls.assert_called_once_with(region_name="us-east-1", profile_name="test-profile")


# ===========================================================================
# AWS — get_session with role assumption
# ===========================================================================


class TestGetSessionWithRoleAssumption:
    @patch("strands_env.utils.aws.boto3.client")
    @patch("botocore.session.get_session")
    def test_assumes_role(self, mock_get_session, mock_boto3_client):
        from datetime import datetime, timezone

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA_TEST",
                "SecretAccessKey": "secret_test",
                "SessionToken": "token_test",
                "Expiration": datetime.now(timezone.utc),
            }
        }
        mock_boto3_client.return_value = mock_sts

        mock_botocore_session = MagicMock()
        mock_get_session.return_value = mock_botocore_session

        role_arn = "arn:aws:iam::123456789:role/TestRole"
        session = get_session(region="us-east-1", role_arn=role_arn)

        mock_boto3_client.assert_called_with("sts", region_name="us-east-1")
        mock_sts.assume_role.assert_called_with(RoleArn=role_arn, RoleSessionName="strands-env")
        assert session is not None

    @patch("strands_env.utils.aws.boto3.client")
    def test_has_refreshable_credentials(self, mock_boto3_client):
        from datetime import datetime, timedelta, timezone

        from botocore.credentials import RefreshableCredentials

        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "AKIA_TEST",
                "SecretAccessKey": "secret_test",
                "SessionToken": "token_test",
                "Expiration": datetime.now(timezone.utc) + timedelta(hours=1),
            }
        }
        mock_boto3_client.return_value = mock_sts

        role_arn = "arn:aws:iam::123456789:role/TestRole"
        session = get_session(role_arn=role_arn)

        botocore_creds = session._session._credentials
        assert isinstance(botocore_creds, RefreshableCredentials)
        assert botocore_creds._refresh_using is not None


# ===========================================================================
# AWS — get_client (cached)
# ===========================================================================


class TestGetClient:
    def setup_method(self):
        get_client.cache_clear()

    @patch("strands_env.utils.aws.boto3.Session")
    def test_returns_client(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        client = get_client("s3", region="us-east-1")
        assert client is mock_session.client.return_value
        mock_session.client.assert_called_once_with("s3", region_name="us-east-1", config=None)

    @patch("strands_env.utils.aws.boto3.Session")
    def test_cached_by_service_and_region(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        client1 = get_client("s3", region="us-east-1")
        client2 = get_client("s3", region="us-east-1")
        assert client1 is client2
        assert mock_session_cls.call_count == 1

    @patch("strands_env.utils.aws.boto3.Session")
    def test_with_boto_config(self, mock_session_cls):
        from botocore.config import Config

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        boto_config = Config(max_pool_connections=20, retries={"max_attempts": 3})
        client = get_client("s3", region="us-east-1", config=boto_config)
        assert client is mock_session.client.return_value
        mock_session.client.assert_called_once_with("s3", region_name="us-east-1", config=boto_config)


# ===========================================================================
# AWS — check_credentials
# ===========================================================================


class TestCheckCredentials:
    def test_returns_true_when_valid(self):
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.return_value = {"Account": "123456789"}
        assert check_credentials(mock_session) is True
        mock_session.client.assert_called_once_with("sts")

    def test_returns_false_on_exception(self):
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.side_effect = Exception("NoCredentials")
        assert check_credentials(mock_session) is False


# ===========================================================================
# Decorators — @requires_env
# ===========================================================================


class TestRequiresEnv:
    async def test_missing_env_var_returns_error_string(self):
        @requires_env("DEFINITELY_NOT_SET_ABC123")
        async def my_tool(self, query: str) -> str:
            return "result"

        result = await my_tool(None, "test")
        assert "DEFINITELY_NOT_SET_ABC123" in result
        assert "Error" in result

    async def test_all_vars_present_calls_function(self):
        @requires_env("PATH")  # PATH is always set
        async def my_tool(self, query: str) -> str:
            return "success"

        result = await my_tool(None, "test")
        assert result == "success"

    async def test_multiple_missing_vars_listed(self, monkeypatch):
        @requires_env("MISSING_VAR_ONE", "MISSING_VAR_TWO")
        async def my_tool(self, query: str) -> str:
            return "result"

        monkeypatch.delenv("MISSING_VAR_ONE", raising=False)
        monkeypatch.delenv("MISSING_VAR_TWO", raising=False)

        result = await my_tool(None, "test")
        assert "MISSING_VAR_ONE" in result
        assert "MISSING_VAR_TWO" in result


# ===========================================================================
# Decorators — @cache_by
# ===========================================================================


class TestCacheBy:
    def test_caches_by_specified_args(self):
        call_count = 0

        @cache_by("a", "b")
        def fn(a, b, c=None):
            nonlocal call_count
            call_count += 1
            return (a, b, c)

        r1 = fn(1, 2, c="first")
        r2 = fn(1, 2, c="second")
        assert r1 is r2
        assert call_count == 1

    def test_unhashable_excluded_args(self):
        """Non-key args can be unhashable (dicts, lists) without breaking cache."""

        @cache_by("name")
        def fn(name, config=None):
            return name

        r1 = fn("x", config={"retries": {"max_attempts": 3}})
        r2 = fn("x", config=[1, 2, 3])
        assert r1 is r2

    def test_positional_and_keyword_produce_same_key(self):
        call_count = 0

        @cache_by("a", "b")
        def fn(a, b="default"):
            nonlocal call_count
            call_count += 1
            return a

        fn("x", "y")
        fn("x", b="y")
        fn(a="x", b="y")
        assert call_count == 1


# ===========================================================================
# Decorators — @with_timeout
# ===========================================================================


class TestWithTimeout:
    def test_returns_result_within_timeout(self):
        @with_timeout(5)
        def fast():
            return 42

        assert fast() == 42

    def test_raises_timeout_error(self):
        @with_timeout(1)
        def slow():
            time.sleep(10)

        start = time.time()
        try:
            slow()
            raise AssertionError("Should have raised TimeoutError")
        except TimeoutError as e:
            assert "1 seconds" in str(e)
        elapsed = time.time() - start
        assert elapsed < 3, f"Took {elapsed:.1f}s, should return in ~1s"

    def test_propagates_exception(self):
        @with_timeout(5)
        def raises():
            raise ValueError("boom")

        try:
            raises()
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert str(e) == "boom"

    def test_none_timeout_no_wrapper(self):
        @with_timeout(None)
        def fn():
            return "no timeout"

        assert fn() == "no timeout"

    def test_thread_is_interrupted(self):
        """Verify the timed-out thread stops running (not leaked)."""
        counter = {"value": 0}

        @with_timeout(1)
        def infinite():
            while True:
                counter["value"] += 1
                time.sleep(0.1)

        with pytest.raises(TimeoutError):
            infinite()

        time.sleep(2)
        snapshot = counter["value"]
        time.sleep(1)
        assert counter["value"] == snapshot, "Thread should have been interrupted"

    def test_works_from_non_main_thread(self):
        """The whole point: with_timeout must work outside the main thread."""
        result = {}

        @with_timeout(1)
        def slow():
            time.sleep(10)

        def run():
            try:
                slow()
            except TimeoutError:
                result["ok"] = True

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        assert result.get("ok"), "Timeout should work from non-main thread"
