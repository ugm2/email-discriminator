import hashlib
import io
import os
from datetime import datetime
from typing import List, Tuple

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from pandas import DataFrame
from prefect import flow, get_run_logger, task

from email_discriminator.core.data_fetcher import (
    EmailDatasetBuilder,
    EmailFetcher,
    TLDRContentParser,
)
from email_discriminator.core.data_versioning import GCSVersionedDataHandler

# Fetching configurations from environment variables
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://35.206.147.175:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/")
MODEL_NAME = os.getenv("MODEL_NAME", "email_discriminator")
BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")

mlflow.set_tracking_uri(MLFLOW_URI)


@task
def fetch_unread_emails(builder: EmailDatasetBuilder) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch unread emails and return them as a DataFrame.
    """
    logger = get_run_logger()
    logger.info("Fetching unread emails.")
    df, email_ids = builder.create_predict_dataframe(
        "from:dan@tldrnewsletter.com is:unread"
    )
    logger.info(f"Fetched unread emails with shape {df.shape}")
    return df, email_ids


@task
def upload_unread_emails(df: DataFrame, gcs_handler: GCSVersionedDataHandler) -> str:
    """
    Upload unread emails to Google Cloud Storage and return the timestamp.
    """
    logger = get_run_logger()
    logger.info("Uploading unread emails to Google Cloud Storage.")

    # Calculate the hash of the data content.
    data_hash = hashlib.sha256(df.to_string().encode()).hexdigest()[:10]

    # Convert the DataFrame to a CSV string.
    csv_string = df.to_csv(index=False)

    # Upload the CSV string to GCS.
    gcs_handler.upload_unlabelled_data(csv_string, data_hash)

    logger.info(f"Uploaded unread emails to GCS with data_hash {data_hash}.")
    return data_hash


@task
def delete_emails(email_fetcher: EmailFetcher, email_ids: List[str]) -> None:
    """
    Deletes emails using the provided EmailFetcher.

    Args:
        email_fetcher: An EmailFetcher object.
        email_ids: A list of email IDs.
    """
    logger = get_run_logger()
    logger.info("Deleting emails")
    email_fetcher.delete_emails(email_ids)
    logger.info("Deleted emails")


@task
def load_unlabelled_data(
    gcs_handler: GCSVersionedDataHandler, data_hash: str
) -> DataFrame:
    """
    Load data from a CSV file in Google Cloud Storage.
    """
    logger = get_run_logger()
    logger.info(f"Loading data from unlabelled data with data_hash {data_hash}")

    # Download the CSV string from GCS.
    csv_string = gcs_handler.download_unlabelled_data(data_hash)

    # Load the CSV string into a DataFrame.
    df = pd.read_csv(io.StringIO(csv_string))

    logger.info(f"Loaded data with shape {df.shape}")
    return df


@task
def load_pipeline(model_name: str) -> PythonModel:
    """
    Load a model pipeline from MLFlow.
    """
    logger = get_run_logger()
    logger.info(f"Loading model {model_name}")
    model_uri = (
        f"models:/{model_name}/Production"  # loading the model in 'Production' stage
    )
    pipeline = mlflow.pyfunc.load_model(model_uri)
    return pipeline


@task
def predict(pipeline: PythonModel, data: DataFrame) -> pd.Series:
    """
    Make predictions using a model pipeline and data.
    """
    logger = get_run_logger()
    logger.info("Making predictions")
    # Add the predictions to the DataFrame as a new column.
    data["predicted_is_relevant"] = pipeline.predict(data)
    logger.info(data)
    return data


@task
def upload_predicted_data(df: DataFrame, gcs_handler: GCSVersionedDataHandler) -> str:
    """
    Upload the predictions to Google Cloud Storage.
    """
    logger = get_run_logger()
    logger.info("Uploading predictions to Google Cloud Storage")

    # Calculate the hash of the data content.
    data_hash = hashlib.sha256(df.to_string().encode()).hexdigest()[:10]

    # Convert the DataFrame to a CSV string.
    csv_string = df.to_csv(index=False)

    # Upload the CSV string to GCS.
    gcs_handler.upload_predicted_data(csv_string, data_hash, "new/")

    logger.info(f"Uploaded predictions to GCS with data_hash {data_hash}.")
    return data_hash


@flow(name="predict-flow")
def predict_flow(do_delete_emails: bool) -> None:
    """
    The main flow for fetching emails, loading data, loading a model, and making predictions.
    """
    logger = get_run_logger()
    logger.info("Starting prediction flow")

    # Create a GCSVersionedDataHandler instance.
    gcs_handler = GCSVersionedDataHandler(BUCKET_NAME)

    # Create a EmailDatasetBuilder instance.
    email_fetcher = EmailFetcher()
    builder = EmailDatasetBuilder(email_fetcher, TLDRContentParser())

    # Fetch unread emails and upload them to GCS.
    unread_emails, email_ids = fetch_unread_emails(builder)
    data_hash = upload_unread_emails(unread_emails, gcs_handler)

    # Delete emails from the user's Gmail account.
    if do_delete_emails:
        delete_emails(email_fetcher, email_ids)

    # Load the unlabelled data.
    df = load_unlabelled_data(gcs_handler, data_hash)

    # Load the model pipeline.
    pipeline = load_pipeline(MODEL_NAME)

    # Make predictions.
    predicted_data = predict(pipeline, df)

    # Upload the predictions to GCS.
    data_hash = upload_predicted_data(predicted_data, gcs_handler)


if __name__ == "__main__":
    # For test purposes we don't delete emails from the user's Gmail account.
    predict_flow(do_delete_emails=False)
