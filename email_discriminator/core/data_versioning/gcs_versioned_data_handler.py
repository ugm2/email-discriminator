import logging
import os
import pickle
from typing import Optional

from google.cloud import storage
from rich.logging import RichHandler

# Setting up logging
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("GCSVersionedDataHandler")


class GCSVersionedDataHandler:
    def __init__(self, bucket_name, credentials=None):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(credentials=credentials)

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

    def move_file_string(self, gcs_file_path: str, new_gcs_file_path: str):
        logger.info(f"Moving file from {gcs_file_path} to {new_gcs_file_path}...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(gcs_file_path)
        blob.move_to(bucket.blob(new_gcs_file_path))
        logger.info(f"File moved from {gcs_file_path} to {new_gcs_file_path}.")

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

    def upload_predicted_data(self, csv_string: str, data_hash: str, folder: str):
        logger.info("Uploading predicted data...")
        self.upload_string(
            csv_string, f"data/predicted_data/{folder}tldr_articles_{data_hash}.csv"
        )
        logger.info("Predicted data uploaded.")

    def download_predicted_data(self, data_hash: str):
        logger.info("Downloading predicted data...")
        data = self.download_string(
            f"data/predicted_data/new/tldr_articles_{data_hash}.csv"
        )
        logger.info("Predicted data downloaded.")
        return data

    def move_predicted_data_to_old(self, data_hash: str):
        logger.info("Moving predicted data to old data...")
        self.move_file_string(
            f"data/predicted_data/new/tldr_articles_{data_hash}.csv",
            f"data/predicted_data/old/tldr_articles_{data_hash}.csv",
        )
        logger.info("Predicted data moved to old data.")

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

    def download_new_predicted_data(self):
        logger.info("Downloading new predicted data...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix="data/predicted_data/new/")

        predicted_data_files = {}

        for blob in blobs:
            # Extract the data hash from the file name
            file_name = blob.name
            logging.debug(f"Downloading file {file_name}")
            data_hash = file_name.split("_")[-1].replace(".csv", "")

            # If the blob is not a file, skip it
            if not file_name.endswith(".csv"):
                continue

            # Download the file as a string and add to the dictionary
            predicted_data_files[data_hash] = blob.download_as_text()

        logger.info("New predicted data downloaded.")
        return predicted_data_files

    def delete_predicted_file(self, data_hash: str):
        logger.info(f"Deleting file {data_hash}...")
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(f"data/predicted_data/new/tldr_articles_{data_hash}.csv")
        blob.delete()
        logger.info(f"File {data_hash} deleted.")

    def read_token_from_gcs(self, bucket_name: str, blob_name: str):
        """Reads token from a GCS bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return pickle.loads(blob.download_as_bytes())

    def write_token_to_gcs(self, creds, bucket_name: str, blob_name: str):
        """Writes token to a GCS bucket."""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pickle.dumps(creds))
