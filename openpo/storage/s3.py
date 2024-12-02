import json
import os
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError


class S3Storage:
    """Storage adapter for Amazon S3.

    This class provides methods to store and retrieve data from Amazon S3 buckets.
    It handles JSON serialization/deserialization and manages S3 operations through
    boto3 client.

    Parameters:
        **kwargs: boto3 S3 client instance configured with provided credentials: region_name, aws_access_key_id, aws_secret_access_key, profile_name

    Raises:
        ClientError: If S3 initialization fails.
    """

    def __init__(self, **kwargs):
        self.s3 = boto3.client("s3", **kwargs)

    def _read_file(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        try:
            res = self.s3.get_object(Bucket=bucket, Key=key)

            content = res["Body"].read()
            data = json.loads(content)

            if isinstance(data, list):
                return data
            return list(data)
        except ClientError as err:
            raise err

    def push_to_s3(
        self,
        data: List[Dict[str, Any]],
        bucket: str,
        key: str = None,
    ):
        """Upload data to an S3 bucket.

        Args:
            data (List[Dict[str, Any]]): The data to upload, will be serialized to JSON.
            bucket (str): Name of the S3 bucket.
            key (str, optional): Object key (path) in the bucket.

        Raises:
            ClientError: If S3 operation fails.
        """
        try:
            json_str = json.dumps(data, default="str")

            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_str,
                ContentType="application/json",
            )

        except ClientError as err:
            raise err

    def read_from_s3(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        """Read data from an S3 bucket.

        Args:
            bucket (str): Name of the S3 bucket.
            key (str): Object key (path) in the bucket.

        Returns:
            List[Dict[str, Any]]: The loaded data as a list of dictionaries.

        Raises:
            ClientError: If S3 operation fails.
        """
        content = self._read_file(bucket, key)
        return content
