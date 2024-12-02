# Storage Providers

OpenPO is integrated with S3 and Hugging Face Hub out of the box. Use storage class to easily upload and download datasets.

## Usage
```python
hf_storage = HuggingFaceStorage(repo_id="my-dataset-repo", api_key="hf-token")

# Save data
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
hf_storage.push_to_hub(data=preference, filename="my-data.json")

# Load data
data = hf_storage.load_from_hub(filename="my-data.json")
```

## Hugging Face Storage

```python
from openpo.providers.huggingface import HuggingFaceStorage

storage = HuggingFaceStorage(
    repo_id="username/repo_name",  # The repository ID on HuggingFace
    api_key="hf-token"            # HuggingFace API token with write access
)
```

### Methods
- `push_to_hub(data: Dict[str, Any], filename: str) -> bool`
    - Parameters:
        - `data`: Dictionary containing the data to save
        - `filename`: Name of the file to save the data to

- `load_from_hub(filename: str) -> Dict[str, Any]`
    - Parameters:
        - `filename`: Name of the file to load



## S3 Storage

```python
from openpo.providers.s3 import S3Storage

storage = S3Storage(
    region_name="us-west-2",              # Optional: AWS region
    aws_access_key_id="access_key",       # Optional: AWS access key
    aws_secret_access_key="secret_key",   # Optional: AWS secret key
    profile_name="default"                # Optional: AWS profile name
)
```

### Methods
- `push_to_s3(data: List[Dict[str, Any]], bucket: str, key: str = None)`
    - Parameters:
        - `data`: List of dictionaries containing the data to save
        - `bucket`: Name of the S3 bucket
        - `key`: Object key (path) in the bucket

- `load_from_s3(bucket: str, key: str) -> List[Dict[str, Any]]`
    - Parameters:
        - `bucket`: Name of the S3 bucket
        - `key`: Object key (path) in the bucket

