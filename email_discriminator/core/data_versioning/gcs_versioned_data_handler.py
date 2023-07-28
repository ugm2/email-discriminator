import logging
import os
from typing import Optional

from google.cloud import storage
from rich.logging import RichHandler

# Setting up logging
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("GCSVersionedDataHandler")


class GCSVersionedDataHandler:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()

    def enable_versioning(self):
        logger.info("Enabling versioning...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        bucket.versioning_enabled = True
        bucket.patch()
        logger.info("Versioning enabled.")

    def disable_versioning(self):
        logger.info("Disabling versioning...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        bucket.versioning_enabled = False
        bucket.patch()
        logger.info("Versioning disabled.")

    def upload_string(self, string_data: str, gcs_file_path: str):
        logger.info(f"Uploading data to {gcs_file_path}...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(gcs_file_path)
        blob.upload_from_string(string_data)
        logger.info(f"Data uploaded to {gcs_file_path}.")

    def download_string(self, gcs_file_path: str, version: Optional[int] = None) -> str:
        logger.info(f"Downloading data from {gcs_file_path}...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(gcs_file_path, generation=version)
        data = blob.download_as_text()
        logger.info(f"Data downloaded from {gcs_file_path}.")
        return data

    def upload_original_data(self, local_file_path: str):
        logger.info("Uploading original data...")
        self.upload_string(local_file_path, "data/original_data/tldr_articles.csv")
        logger.info("Original data uploaded.")

    def download_original_data(self):
        logger.info("Downloading original data...")
        data = self.download_string("data/original_data/tldr_articles.csv")
        logger.info("Original data downloaded.")
        return data

    def upload_unlabelled_data(self, csv_string: str, data_hash: str):
        logger.info("Uploading unlabelled data...")
        self.upload_string(
            csv_string, f"data/unlabelled_data/tldr_articles_{data_hash}.csv"
        )
        logger.info("Unlabelled data uploaded.")

    def download_unlabelled_data(self, data_hash: str):
        logger.info("Downloading unlabelled data...")
        data = self.download_string(
            f"data/unlabelled_data/tldr_articles_{data_hash}.csv"
        )
        logger.info("Unlabelled data downloaded.")
        return data

    def upload_predicted_data(self, csv_string: str, data_hash: str):
        logger.info("Uploading predicted data...")
        self.upload_string(
            csv_string, f"data/predicted_data/tldr_articles_{data_hash}.csv"
        )
        logger.info("Predicted data uploaded.")

    def download_predicted_data(self, data_hash: str):
        logger.info("Downloading predicted data...")
        data = self.download_string(
            f"data/predicted_data/tldr_articles_{data_hash}.csv"
        )
        logger.info("Predicted data downloaded.")
        return data

    def upload_training_data(self, csv_string: str, data_hash: str):
        logger.info("Uploading training data...")
        self.upload_string(
            csv_string, f"data/training_data/tldr_articles_{data_hash}.csv"
        )
        logger.info("Training data uploaded.")

    def download_training_data(self, data_hash: str):
        logger.info("Downloading training data...")
        data = self.download_string(f"data/training_data/tldr_articles_{data_hash}.csv")
        logger.info("Training data downloaded.")
        return data

    def download_all_training_data(self):
        logger.info("Downloading all training data...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix="data/training_data/")

        training_data_files = {}

        for blob in blobs:
            # Extract the data hash from the file name
            file_name = blob.name
            logging.debug(f"Downloading file {file_name}")
            data_hash = file_name.split("_")[-1].replace(".csv", "")

            # If the blob is not a file, skip it
            if not file_name.endswith(".csv"):
                continue

            # Download the file as a string and add to the dictionary
            training_data_files[data_hash] = blob.download_as_text()

        logger.info("All training data downloaded.")
        return training_data_files


if __name__ == "__main__":
    BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")
    gcs_handler = GCSVersionedDataHandler(BUCKET_NAME)
    training_data_files = gcs_handler.download_all_training_data()
