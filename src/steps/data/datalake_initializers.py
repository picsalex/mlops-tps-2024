from typing import List

from zenml import step
from zenml.logger import get_logger

from src.config.settings import (
    MINIO_DATA_SOURCES_BUCKET_NAME,
    MINIO_DATASETS_BUCKET_NAME,
    MINIO_ENDPOINT,
    MINIO_PENDING_ANNOTATIONS_BUCKET_NAME,
    MINIO_PENDING_REVIEWS_BUCKET_NAME,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
)
from src.materializers.materializer_bucket_client import BucketClientMaterializer
from src.materializers.materializer_data_source import DataSourceMaterializer
from src.models.model_bucket_client import BucketClient, MinioClient
from src.models.model_data_source import DataSourceList, HuggingFaceDataSource


@step
def validate_bucket_connection(bucket_client: BucketClient) -> None:
    """
    Validate the connection to the bucket_client.

    Args:
        bucket_client (BucketClient): The bucket Client to test the connection.

    Raises:
        ConnectionError: If the connection to the bucket_client fails.
    """
    logger = get_logger(__name__)

    try:
        bucket_client.check_connection()
        logger.info("Connection to the bucket Client verified successfully.")
    except Exception as e:
        logger.error(
            f"Connection to the storage service failed: {type(e).__name__}: {e}"
        )
        raise


@step
def setup_bucket(
    bucket_client: BucketClient, bucket_name: str, enable_versioning: bool
) -> None:
    """
    Set up a bucket in the provided bucket_client. Creates the bucket if it does not exist and enables versioning if specified.

    Args:
        bucket_client (BucketClient): The bucket client used for bucket operations.
        bucket_name (str): The name of the bucket to create or configure.
        enable_versioning (bool): Flag to enable versioning on the bucket.
    """
    logger = get_logger(__name__)

    if not bucket_client.bucket_exists(bucket_name):
        bucket_client.make_bucket(bucket_name, enable_versioning)
        logger.info(
            f"Successfully created bucket {bucket_name} (versioning set to"
            f" {enable_versioning})"
        )
    else:
        logger.warning(f"The bucket {bucket_name} already exists. Skipping.")


@step(output_materializers=BucketClientMaterializer)
def minio_client_initializer() -> MinioClient:
    return MinioClient(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ROOT_USER,
        secret_key=MINIO_ROOT_PASSWORD,
        secure=False,
    )


@step
def bucket_name_list_initializer() -> List[str]:
    """
    Retrieve a list of bucket names to be created the bucket client.

    Returns:
        List[str]: A list of bucket names for creation and management.
    """
    return [
        MINIO_PENDING_ANNOTATIONS_BUCKET_NAME,
        MINIO_PENDING_REVIEWS_BUCKET_NAME,
        MINIO_DATA_SOURCES_BUCKET_NAME,
        MINIO_DATASETS_BUCKET_NAME,
    ]


@step(output_materializers=DataSourceMaterializer)
def data_source_list_initializer() -> DataSourceList:
    """
    Retrieve a list of DataSource. Those DataSource will then be imported into the Datalake.

    Returns:
        DataSourceList: A list of DataSource.
    """
    return DataSourceList(
        [
            HuggingFaceDataSource(
                dataset_name="kili-technology/plastic_in_river",
                label_map={
                    # class_id: "class_name",
                    0: "PLASTIC_BAG",
                    1: "PLASTIC_BOTTLE",
                    2: "OTHER_PLASTIC_WASTE",
                    3: "NOT_PLASTIC_WASTE",
                },
            )
        ]
    )


@step
def datalake_initializer(
    bucket_client: BucketClient,
    bucket_name_list: list[str],
    enable_versioning: bool = True,
) -> None:
    """
    ZenML pipeline to prepare the buckets for data storage and management.
    This flow creates and configures buckets as needed and enables versioning.
    """
    validate_bucket_connection(bucket_client)

    for bucket_name in bucket_name_list:
        setup_bucket(
            bucket_client=bucket_client,
            bucket_name=bucket_name,
            enable_versioning=enable_versioning,
        )
