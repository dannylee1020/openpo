import json
import os
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError


class S3Storage:
    def __init__(self, **kwargs):
        """
        Args:
            bucket_name: Name of the S3 bucket
            **kwargs: Optional arguments passed to boto3.client:
                - region_name: AWS region
                - aws_access_key_id: AWS access key
                - aws_secret_access_key: AWS secret key
                - profile_name: AWS profile name
        """
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

    def push_to_hub(
        self,
        data: List[Dict[str, Any]],
        bucket: str,
        key: str = None,
    ):
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

    def read_from_hub(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        content = self._read_file(bucket, key)
        return content
