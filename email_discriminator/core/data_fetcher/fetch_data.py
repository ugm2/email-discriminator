import os

from rich import print

from email_discriminator.core.data_fetcher.content_parser import TLDRContentParser
from email_discriminator.core.data_fetcher.email_dataset_builder import (
    EmailDatasetBuilder,
)
from email_discriminator.core.data_fetcher.email_fetcher import EmailFetcher


def fetch_data():
    fetcher = EmailFetcher()
    parser = TLDRContentParser()

    builder = EmailDatasetBuilder(fetcher, parser)
    articles_df = builder.create_training_dataframe(
        "label:TLDRs", "from:dan@tldrnewsletter.com is:read -label:TLDRs"
    )

    print(articles_df)
    print(articles_df["is_relevant"].value_counts())

    # Save dataframe to csv
    os.makedirs("data", exist_ok=True)
    articles_df.to_csv("data/tldr_articles.csv", index=False)

    # Create dataframe for unread TLDR emails
    unread_df = builder.create_predict_dataframe(
        "from:dan@tldrnewsletter.com is:unread"
    )
    print(unread_df)


if __name__ == "__main__":
    fetch_data()
