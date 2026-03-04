# Copyright 2025-2026 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for AWS session utilities."""

from unittest.mock import MagicMock, patch

import boto3

from strands_env.utils.aws import check_credentials, get_client, get_session


class TestGetSession:
    """Tests for get_session (returns a fresh session each time)."""

    def test_returns_session(self):
        """Should return a boto3 Session."""
        session = get_session(region="us-west-2")
        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    def test_fresh_session_each_call(self):
        """Each call should return a new session (not cached)."""
        session1 = get_session(region="us-east-1")
        session2 = get_session(region="us-east-1")
        assert session1 is not session2

    @patch("strands_env.utils.aws.boto3.Session")
    def test_passes_profile(self, mock_session_cls):
        """Should pass profile_name to boto3.Session."""
        mock_session_cls.return_value = MagicMock()
        get_session(region="us-east-1", profile_name="test-profile")
        mock_session_cls.assert_called_once_with(region_name="us-east-1", profile_name="test-profile")


class TestGetSessionWithRoleAssumption:
    """Tests for get_session with role_arn (role assumption mode)."""

    @patch("strands_env.utils.aws.boto3.client")
    @patch("botocore.session.get_session")
    def test_assumes_role(self, mock_get_session, mock_boto3_client):
        """Should call STS assume_role when role_arn provided."""
        from datetime import datetime, timezone

        # Mock STS response
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

        # Mock botocore session
        mock_botocore_session = MagicMock()
        mock_get_session.return_value = mock_botocore_session

        role_arn = "arn:aws:iam::123456789:role/TestRole"
        session = get_session(region="us-east-1", role_arn=role_arn)

        # Verify STS was called
        mock_boto3_client.assert_called_with("sts", region_name="us-east-1")
        mock_sts.assume_role.assert_called_with(RoleArn=role_arn, RoleSessionName="strands-env")

        # Verify session was created
        assert session is not None

    @patch("strands_env.utils.aws.boto3.client")
    def test_has_refreshable_credentials(self, mock_boto3_client):
        """Session should have RefreshableCredentials with refresh callback."""
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

        # Get the underlying botocore credentials directly
        botocore_creds = session._session._credentials

        # Verify it's RefreshableCredentials with a refresh callback
        assert isinstance(botocore_creds, RefreshableCredentials)
        assert botocore_creds._refresh_using is not None


class TestGetClient:
    """Tests for get_client (cached, thread-safe clients)."""

    def setup_method(self):
        get_client.cache_clear()

    @patch("strands_env.utils.aws.boto3.Session")
    def test_returns_client(self, mock_session_cls):
        """Should return a boto3 client."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        client = get_client("s3", region="us-east-1")
        assert client is mock_session.client.return_value
        mock_session.client.assert_called_once_with("s3", region_name="us-east-1")

    @patch("strands_env.utils.aws.boto3.Session")
    def test_cached_by_service_and_region(self, mock_session_cls):
        """Same service+region should return cached client."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        client1 = get_client("s3", region="us-east-1")
        client2 = get_client("s3", region="us-east-1")
        assert client1 is client2
        # Session should only be created once
        assert mock_session_cls.call_count == 1


class TestCheckCredentials:
    """Tests for check_credentials."""

    def test_returns_true_when_valid(self):
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.return_value = {"Account": "123456789"}
        assert check_credentials(mock_session) is True
        mock_session.client.assert_called_once_with("sts")

    def test_returns_false_on_exception(self):
        mock_session = MagicMock()
        mock_session.client.return_value.get_caller_identity.side_effect = Exception("NoCredentials")
        assert check_credentials(mock_session) is False
