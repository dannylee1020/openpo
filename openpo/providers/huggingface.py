import json
from typing import Any, Dict, List

from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_file


class HuggingFaceStorage:
    def __init__(self, repo_id: str, api_key: str):
        """Initialize HuggingFace adapter with repository ID and access token.

        Args:
            repo_id (str): The repository ID on HuggingFace (format: 'username/repo_name')
            token (str): HuggingFace API token with write access
        """
        self.repo_id = repo_id
        self.api = HfApi(token=api_key)

        try:
            create_repo(repo_id, token=api_key, repo_type="dataset", exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to initialize HuggingFace repository: {str(e)}")

    def save_data(self, data: Dict[str, Any], filename: str) -> bool:
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

            return True
        except Exception as e:
            print(f"Error saving data to HuggingFace: {str(e)}")
            return False

    def load_data(self, filename: str) -> Dict[str, Any]:
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id, filename=filename, repo_type="dataset"
            )

            with open(local_path, "r") as f:
                return json.loads(f.read())
        except Exception as e:
            print(f"Error loading data from HuggingFace: {str(e)}")
            return {}

    def load_data_all(self) -> List[Dict[str, Any]]:
        try:
            files = self.api.list_repo_files(self.repo_id, repo_type="dataset")
            json_files = [f for f in files if f.endswith(".json")]

            all_data = []
            for file in json_files:
                local_path = hf_hub_download(
                    repo_id=self.repo_id, filename=file, repo_type="dataset"
                )

                with open(local_path, "r") as f:
                    data = json.loads(f.read())
                    all_data.extend(data)
            return all_data
        except Exception as e:
            print(f"Error loading all data from HuggingFace: {str(e)}")
            return {}
