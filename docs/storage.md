# Storage Providers

OpenPO is integrated with S3 and HuggingFace Hub out of the box. Use providers to easily upload and download datasets.

## HuggingFace Storage

### Initialization
```python
from openpo.providers.huggingface import HuggingFaceStorage

storage = HuggingFaceStorage(
    repo_id="username/repo_name",  # The repository ID on HuggingFace
    api_key="hf-token"            # HuggingFace API token with write access
)
```

### Methods
- `save_data(data: Dict[str, Any], filename: str) -> bool`
    - Parameters:
        - `data`: Dictionary containing the data to save
        - `filename`: Name of the file to save the data to

- `load_data(filename: str) -> Dict[str, Any]`
    - Parameters:
        - `filename`: Name of the file to load

- `load_data_all() -> List[Dict[str, Any]]`


## S3 Storage

### Initialization
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
- `save_data(data: List[Dict[str, Any]], bucket: str, key: str = None)`
    - Parameters:
        - `data`: List of dictionaries containing the data to save
        - `bucket`: Name of the S3 bucket
        - `key`: Object key (path) in the bucket

- `load_data(bucket: str, key: str) -> List[Dict[str, Any]]`
    - Parameters:
        - `bucket`: Name of the S3 bucket
        - `key`: Object key (path) in the bucket

- `load_data_all(bucket: str, limit: int) -> List[Dict[str, Any]]`
    - Parameters:
        - `bucket`: Name of the S3 bucket
        - `limit`: Maximum number of files to read

### Example Usage
```python
storage = HuggingFaceStorage(repo_id="my-dataset-repo", api_key="hf-token")

# Save data
preference = {"prompt": "text", "preferred": "response1", "rejected": "response2"}
storage.save_data(data=preference, filename="my-data.json")

# Load data
data = storage.load_data(filename="my-data.json")
print(data)
```