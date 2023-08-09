import hashlib
import os

import pandas as pd

from email_discriminator.core.data_versioning import GCSVersionedDataHandler

# Initialize GCS handler
BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")
gcs_handler = GCSVersionedDataHandler(BUCKET_NAME)


def download_data() -> pd.DataFrame:
    """
    Downloads new predicted data from the GCS bucket.
    """
    return gcs_handler.download_new_predicted_data()


def split_dataframe(df: pd.DataFrame, end_index: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataframe into "reviewed" and "unreviewed" datasets.
    """
    return df.iloc[:end_index], df.iloc[end_index:]


def upload_reviewed_data(
    reviewed_df: pd.DataFrame, unreviewed_df: pd.DataFrame, old_file_name: str
):
    """
    Uploads reviewed and unreviewed data to GCS, and deletes the old file.
    """
    # Convert the dataframes to CSV strings
    reviewed_csv_string = reviewed_df.to_csv(index=False)
    unreviewed_csv_string = unreviewed_df.to_csv(index=False)

    # Compute the data hash for each CSV string
    reviewed_data_hash = hashlib.sha256(reviewed_df.to_string().encode()).hexdigest()[
        :10
    ]
    unreviewed_data_hash = hashlib.sha256(
        unreviewed_df.to_string().encode()
    ).hexdigest()[:10]

    # Upload the reviewed data to the training_data folder
    gcs_handler.upload_training_data(reviewed_csv_string, reviewed_data_hash)

    # Upload the unreviewed data to the 'new' folder and the reviewed data to the 'old' folder in the predicted_data directory
    gcs_handler.upload_predicted_data(reviewed_csv_string, reviewed_data_hash, "old/")
    gcs_handler.upload_predicted_data(
        unreviewed_csv_string, unreviewed_data_hash, "new/"
    )

    # Delete the old file from the 'new' folder
    gcs_handler.delete_predicted_file(old_file_name)
