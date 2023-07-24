import logging
import os

import pandas as pd
from rich.logging import RichHandler

from email_discriminator.core.data_fetcher.content_parser import ContentParserInterface
from email_discriminator.core.data_fetcher.email_fetcher import EmailFetcher

# Get the logger level from environment variables. Default to WARNING if not set.
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("EmailDatasetBuilder")


class EmailDatasetBuilder:
    """Class to build datasets from email data."""

    def __init__(self, fetcher: EmailFetcher, parser: ContentParserInterface):
        """
        Constructor for EmailDatasetBuilder.

        Args:
        fetcher: An instance of the EmailFetcher class.
        parser: An instance of a class that implements the ContentParserInterface.
        """
        self.fetcher = fetcher
        self.parser = parser

    def create_training_dataframe(
        self, relevant_query: str, irrelevant_query: str
    ) -> pd.DataFrame:
        """
        Creates a training dataframe with labeled data.

        Args:
        relevant_query: Query string to fetch relevant emails.
        irrelevant_query: Query string to fetch irrelevant emails.

        Returns:
        A DataFrame containing the relevant and irrelevant articles.
        """
        try:
            relevant_emails = self.fetcher.fetch_emails(relevant_query)
            irrelevant_emails = self.fetcher.fetch_emails(irrelevant_query)
        except Exception as e:
            logger.error(f"Failed to fetch emails due to {str(e)}")
            raise

        try:
            relevant_articles = self.fetcher.get_articles_from_emails(
                relevant_emails, self.parser.parse_content
            )
            irrelevant_articles = self.fetcher.get_articles_from_emails(
                irrelevant_emails, self.parser.parse_content
            )
        except Exception as e:
            logger.error(f"Failed to get articles from emails due to {str(e)}")
            raise

        relevant_df = pd.DataFrame(relevant_articles)
        relevant_df["is_relevant"] = 1

        irrelevant_df = pd.DataFrame(irrelevant_articles)
        irrelevant_df["is_relevant"] = 0

        return pd.concat([relevant_df, irrelevant_df], ignore_index=True)

    def create_predict_dataframe(self, query: str) -> pd.DataFrame:
        """
        Creates a prediction dataframe with unlabeled data.

        Args:
        query: Query string to fetch emails.

        Returns:
        A DataFrame containing articles extracted from the fetched emails.
        """
        try:
            unread_emails = self.fetcher.fetch_emails(query)
        except Exception as e:
            logger.error(f"Failed to fetch emails due to {str(e)}")
            raise

        try:
            articles = self.fetcher.get_articles_from_emails(
                unread_emails, self.parser.parse_content
            )
        except Exception as e:
            logger.error(f"Failed to get articles from emails due to {str(e)}")
            raise

        return pd.DataFrame(articles)
