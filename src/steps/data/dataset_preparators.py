from zenml import step
# from src.models.model_dataset import (to_yolo_format)
from src.models.model_dataset import Dataset
# from src.clients.minio_client import MinioClient
from src.models.model_bucket_client import BucketClient, MinioClient
from src.models.model_data_source import DataSourceList
import os
from src.config.settings import EXTRACTED_DATASETS_PATH

@step
def dataset_creator():
    pass

@step
def datasource_extractor(data_source_list: DataSourceList, minio_client: MinioClient, bucket_name: str):
    # data_source_list.data_sources[0] est le seul dataset qu'on a sur le datalake
    data_source = data_source_list.data_sources[0]
    # On télécharge le dataset dans le dossier destination_folder
    print(data_source.name)
    minio_client.download_folder(bucket_name, data_source.name, EXTRACTED_DATASETS_PATH)

    # # Pour regarder ce qu'il y a dans data_source_list
    # for data_source in data_source_list.data_sources:
    #     print(data_source)

    # # Faut que destination_path soit ./destination_folder
    # destination_path = os.path.join(os.path.basename("./"), destination_folder)
    

    

@step
def dataset_to_yolo_converter(dataset : Dataset, dataset_path: str):
    return dataset.to_yolo_format(dataset, dataset_path)
