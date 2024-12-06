import io
import json
from typing import Any, Dict, List, Literal, Optional, Union

import boto3
import pandas as pd
from botocore.exceptions import ClientError


class S3Storage:
    """Storage adapter for Amazon S3.

    This class provides methods to store and retrieve data from Amazon S3 buckets.
    It handles JSON serialization/deserialization and manages S3 operations through
    boto3 client.

    Parameters:
        **kwargs: Keyword arguments can be passed to access AWS:

            - region_name
            - aws_access_key_id
            - aws_secret_access_key
            - profile_name

            Alternatively, credentials can be configured with aws configure

    Raises:
        ClientError: If S3 initialization fails.
    """

    def __init__(self, **kwargs):
        self.s3 = boto3.client("s3", **kwargs)

    def _read_file(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        try:
            res = self.s3.get_object(Bucket=bucket, Key=key)
            content = res["Body"].read()

            file_ext = key.split(".")[-1].lower()
            if file_ext == "json":
                data = json.loads(content)
                if isinstance(data, list):
                    return data
                return list(data)
            elif file_ext == "parquet":
                try:
                    parquet_buffer = io.BytesIO(content)
                    df = pd.read_parquet(parquet_buffer)
                    return json.loads(df.to_json(orient="records"))
                except Exception as e:
                    raise ValueError(f"Failed to parse content as parquet: {str(e)}")
            else:
                raise ValueError(
                    f"Unsupported content type: {content_type}. Supported extensions are: json, parquet "
                )
        except ClientError as err:
            raise err

    def _serialize_data(
        self,
        data,
        serialization_type,
    ) -> tuple[bytes, str]:
        if isinstance(data, bytes):
            return data, "application/octet-stream"

        if serialization_type == "parquet":
            buffer = io.BytesIO()

            if isinstance(data, list):
                if not all(isinstance(item, dict) for item in data):
                    raise TypeError(
                        "All items in list must be dictionaries when using 'parquet' serialization"
                    )
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise TypeError(
                    "Data must be DataFrame or list of dicts when using 'parquet' serialization"
                )

            df.to_parquet(buffer)
            return buffer.getvalue(), "application/octet-stream"

        if serialization_type == "json":
            if isinstance(data, pd.DataFrame):
                data = json.loads(data.to_json(orient="records"))
            elif not isinstance(data, list):
                raise TypeError(
                    "Data must be a list or DataFrame when using 'json' serialization"
                )

            return json.dumps(data, default=str).encode(), "application/json"

        raise ValueError(f"Unsupported serialization type: {serialization_type}")

    def push_to_s3(
        self,
        data: Union[List[Dict[str, Any]], pd.DataFrame, bytes],
        bucket: str,
        key: Optional[str] = None,
        ext_type: Literal["parquet", "json"] = "parquet",
    ):
        """Upload data to an S3 bucket.

        Args:
            data: The data to upload.

                - List[Dict]: List of dictionaries
                - pd.DataFrame: Pandas DataFrame

            bucket (str): Name of the S3 bucket
            key (str, optional): Object key (path) in the bucket

            ext_type (str): Type of serialization to use:

                - parquet: Serialize as parquet
                - json: Serialize as JSON


        Raises:
            ClientError: If S3 operation fails
            TypeError: If data type is not compatible with chosen serialization type
            ValueError: If serialization type is not supported or data cannot be deserialized
        """
        try:
            serialized_data, content_type = self._serialize_data(data, ext_type)

            self.s3.put_object(
                Bucket=bucket,
                Key=f"{key}.{ext_type}",
                Body=serialized_data,
                ContentType=content_type,
            )

        except ClientError as err:
            raise err

    def load_from_s3(self, bucket: str, key: str) -> List[Dict[str, Any]]:
        """Read data from an S3 bucket.

        Args:
            bucket (str): Name of the S3 bucket.
            key (str): Object name (path) in the bucket.

        Returns:
            List[Dict]: The loaded data as a list of dictionaries.

        Raises:
            ClientError: If S3 operation fails.
            ValueError: If content type is not supported or content cannot be parsed.
        """
        content = self._read_file(bucket, key)
        return content
