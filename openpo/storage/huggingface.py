import json
from typing import Any, Dict, List

from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_file


class HuggingFaceStorage:
    """Storage adapter for HuggingFace Datasets.

    This class provides methods to store and retrieve data from HuggingFace's dataset
    repositories. It handles the creation of repositories and manages data upload
    and download operations.

    Parameters:
        repo_id (str): The repository identifier on HuggingFace.
        api_key (HfApi): HuggingFace API token with write access

    Raises:
        Exception: If repository initialization fails.
    """

    def __init__(self, repo_id: str, api_key: str):
        self.repo_id = repo_id
        self.api = HfApi(token=api_key)

        try:
            create_repo(repo_id, token=api_key, repo_type="dataset", exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to initialize HuggingFace repository: {str(e)}")

    def push_to_hub(self, data: Dict[str, Any], filename: str):
        """Upload data to HuggingFace dataset repository.

        Args:
            data (Dict[str, Any]): The data to upload, will be serialized to JSON.
            filename (str): Name of the file to create in the repository.

        Raises:
            Exception: If pushing data to HuggingFace Hub fails

        """
        try:
            data_str = json.dumps(data)

            with open(filename, "w") as f:
                f.write(data_str)

            upload_file(
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="dataset",
            )

        except Exception as e:
            raise Exception(f"error pushing data to Hugging Face Hub: {e}")

    def read_from_hub(self, filename: str) -> Dict[str, Any]:
        """Read data from HuggingFace dataset repository.

        Args:
            filename (str): Name of the file to read from the repository.

        Returns:
            Dict[str, Any]: The loaded data as a dictionary. Returns empty dict if
                reading fails.
        """
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id, filename=filename, repo_type="dataset"
            )

            with open(local_path, "r") as f:
                return json.loads(f.read())
        except Exception as e:
            print(f"Error loading data from HuggingFace: {str(e)}")
            return {}
