from enum import Enum

from huggingface_hub import HfApi


class HuggingFaceRepositoryType(Enum):
    SPACE = "space"


class HuggingFaceClient:
    def __init__(
        self,
        destination_path: str,
        space_repository_id: str,
        repository_type: HuggingFaceRepositoryType,
        access_token: str,
    ):
        self.destination_path = destination_path
        self.space_repository_id = space_repository_id
        self.repository_type = repository_type
        self.access_token = access_token

        self.huggingface_api = HfApi(token=self.access_token)

    def format_commit_message(
        self, model_name: str, model_version: int, model_tag: str
    ):
        return f"Updated {model_name} with model_version {model_version}@{model_tag}"

    def upload_space(self, space_path: str, commit_message: str):
        api = HfApi()
        api.upload_folder(
            folder_path=space_path,
            repo_id=self.space_repository_id,
            repo_type=self.repository_type.value,
            commit_message=commit_message,
        )
