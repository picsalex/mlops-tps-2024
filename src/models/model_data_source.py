import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import requests
import ulid

from src.models.model_datasource_metadata import DataSourceMetadata, DataSourceType


class DataSource(ABC):
    def __init__(
        self,
        root_folder_path: str,
        label_map: dict[int, str],
    ):
        self.root_folder_path = root_folder_path
        self.label_map = label_map
        self.uuid = DataSource.get_data_source_uuid()
        self.name = self.get_data_source_name()

    def get_data_source_name(self) -> str:
        return self.root_folder_path.split("/")[-1]

    @abstractmethod
    def verify_data_source_path(self) -> None:
        pass

    @abstractmethod
    def get_metadata(self) -> DataSourceMetadata:
        pass

    @staticmethod
    def get_data_source_uuid() -> str:
        return str(ulid.new())


class LocalDataSource(DataSource):
    def verify_data_source_path(self) -> None:
        """
        Validate an imported data source's's path.
        Checks if the provided path exists and is a directory. Raises an exception if not.

        Raises:
            FileNotFoundError: If the data source's path does not exist.
            NotADirectoryError: If the data source's path exists but is not a directory.
        """
        if not os.path.exists(self.root_folder_path):
            raise FileNotFoundError(
                f"The data source's path '{self.root_folder_path}' does not exist."
            )

        if not os.path.isdir(self.root_folder_path):
            raise NotADirectoryError(
                f"The data source's '{self.root_folder_path}' is not a directory."
            )

    def get_metadata(self) -> DataSourceMetadata:
        """
        Retrieve metadata information for the local data source.

        Returns:
            DataSourceMetadata: An object containing metadata for the data source.
        """
        return DataSourceMetadata(
            name=self.name,
            uuid=self.uuid,
            source=DataSourceType.IMPORTED,
            creation_date=datetime.now(),
            last_modified_date=datetime.now(),
        )


class HuggingFaceDataSource(DataSource):
    def __init__(
        self,
        dataset_name: str,
        label_map: dict[int, str],
        api_token: str | None = None,
    ):
        super().__init__(root_folder_path=dataset_name, label_map=label_map)
        self.dataset_name = dataset_name
        self.api_token = api_token

    def verify_data_source_path(self) -> None:
        """
        Check if the data source's identifier exists in HuggingFace's Dataset registry.
        Raises specific exceptions based on the dataset's availability and access requirements.

        Raises:
            ValueError: If the dataset is not valid or not found in HuggingFace's registry.
            PermissionError: If the dataset is gated and requires a private API token.
            FileNotFoundError: If there is an issue with the request or the dataset is not found.
        """
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={self.root_folder_path}"

        try:
            response = requests.get(api_url, headers=headers)

            if response.status_code == 401:
                raise PermissionError(
                    "The HuggingFace's dataset is gated and requires a private API"
                    " token for access."
                )
            elif response.status_code == 404:
                raise FileNotFoundError(
                    "There is an issue with the request, or the dataset is not found on"
                    " HuggingFace."
                )
            elif not response.ok:
                raise ValueError(
                    "An error occurred with the request. Status code:"
                    f" {response.status_code}"
                )

            data = response.json()
            if not (data.get("viewer", False) or data.get("preview", False)):
                raise ValueError(
                    f"The dataset '{self.root_folder_path}' is not valid or not"
                    " available on HuggingFace."
                )
        except requests.RequestException as e:
            raise ConnectionError("Failed to connect to HuggingFace server") from e

    def get_metadata(self) -> DataSourceMetadata:
        """
        Retrieve metadata information for the HuggingFace data source.

        Returns:
            DataSourceMetadata: An object containing metadata for the data source.
        """
        return DataSourceMetadata(
            name=self.name,
            uuid=self.uuid,
            source=DataSourceType.HUGGING_FACE,
            creation_date=datetime.now(),
            last_modified_date=datetime.now(),
        )


class DataSourceList:
    def __init__(self, data_sources: List[DataSource]):
        self.data_sources = data_sources
