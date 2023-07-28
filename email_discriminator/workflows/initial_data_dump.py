import os
from datetime import datetime

from prefect import flow, get_run_logger, task

from email_discriminator.core.data_fetcher import (
    EmailDatasetBuilder,
    EmailFetcher,
    TLDRContentParser,
)
from email_discriminator.core.data_versioning.gcs_versioned_data_handler import (
    GCSVersionedDataHandler,
)

BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")


@task(name="Fetch data")
def fetch_data(builder, gcs_handler):
    logger = get_run_logger()
    logger.info("Starting data fetching task.")

    articles_df = builder.create_training_dataframe(
        "label:TLDRs", "from:dan@tldrnewsletter.com is:read -label:TLDRs"
    )

    logger.info("Data fetched, pushing to GCP.")

    # Save the DataFrame to a CSV file.
    file_path = "data/tldr_articles.csv"
    articles_df.to_csv(file_path, index=False)

    # Upload the original data to GCS.
    gcs_handler.upload_original_data(file_path)

    logger.info("Data pushed to GCP.")

    logger.info(articles_df)
    logger.info(articles_df["is_relevant"].value_counts())


@flow(name="fetch_data")
def fetch_data_flow(bucket_name):
    logger = get_run_logger()
    logger.info("Starting data fetching flow.")

    fetcher = EmailFetcher()
    parser = TLDRContentParser()
    builder = EmailDatasetBuilder(fetcher, parser)

    # Create a GCSVersionedDataHandler instance.
    gcs_handler = GCSVersionedDataHandler(bucket_name)

    # Enable versioning on the bucket.
    gcs_handler.enable_versioning()

    # Fetch data and upload to GCS.
    fetch_data(builder, gcs_handler)

    logger.info("Data fetching flow completed.")


if __name__ == "__main__":
    fetch_data_flow(BUCKET_NAME)
