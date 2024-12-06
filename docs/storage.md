# Storage Providers

OpenPO provides storage class for S3 and HuggingFace Dataset repository out of the box. Use storage class to easily upload and download datasets.

## HuggingFace Storage
OpenPO uses HuggingFace `push_to_hub` function under the hood. For more information on the method, refer to the [API Reference section](api.md#storage)

```python
hf_storage = HuggingFaceStorage(repo_id="my-dataset-repo", api_key="hf-token") # api_key can also be set as environment variable.

# push data to repo
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
hf_storage.push_to_repo(data=preference)

# Load data from repo
data = hf_storage.load_from_repo()
```

## S3 Storage
`S3Storage` supports serialization for `json` and `parquet`. To initialize the class, you can either pass in the keyword arguments or configure aws credentials with `aws configure`

```python
from openpo.providers.s3 import S3Storage

s3 = S3Storage(
    region_name="us-west-2",              # Optional: AWS region
    aws_access_key_id="access_key",       # Optional: AWS access key
    aws_secret_access_key="secret_key",   # Optional: AWS secret key
    profile_name="default"                # Optional: AWS profile name
)

# push data to s3
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
s3.push_to_s3(
    data=preference,
    bucket="my-bucket",
    key="my-key",
    ext_type='parquet',
)

# load data from s3
data = s3.load_from_s3(bucket='my-bucket', key='data-key')

```
