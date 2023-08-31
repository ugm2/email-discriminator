import base64
import logging
import os
from typing import Callable, Dict, List

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from rich.logging import RichHandler

from email_discriminator.core.data_versioning import GCSVersionedDataHandler

# Get the logger level from environment variables. Default to WARNING if not set.
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("EmailFetcher")
GCS_BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")


class EmailFetcher:
    """
    A class to interact with the Gmail API, allowing fetching and parsing of emails.

    Attributes:
        creds_path (str): Path to the token pickle file.
        client_secret_path (str): Path to the client secret JSON file.
        service (googleapiclient.discovery.Resource): A Resource object with methods for interacting with the service.
    """

    def __init__(
        self,
        creds_path: str = "secrets/token.pickle",
        client_secret_path: str = "secrets/client_secret.json",
    ):
        """
        Constructs all the necessary attributes for the EmailFetcher object.

        Args:
            creds_path (str): Path to the token pickle file.
            client_secret_path (str): Path to the client secret JSON file.
        """

        self.creds_path = creds_path
        self.client_secret_path = client_secret_path
        self.service = self.get_service()

    def get_service(self) -> Resource:
        """
        Authenticates and returns a service object for the Gmail API.
        """
        gcs_data_handler = GCSVersionedDataHandler(GCS_BUCKET_NAME)
        creds = gcs_data_handler.read_token_from_gcs(GCS_BUCKET_NAME, self.creds_path)

        if not creds or not creds.valid:
            try:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Failed to refresh credentials: {e}")
                creds = None

            if not creds:
                client_secret_json = gcs_data_handler.read_client_secrets_from_gcs(
                    GCS_BUCKET_NAME, self.client_secret_path
                )
                flow = InstalledAppFlow.from_client_config(
                    client_secret_json,
                    ["https://www.googleapis.com/auth/gmail.modify"],
                )
                creds = flow.run_local_server(port=0)
                gcs_data_handler.write_token_to_gcs(
                    creds, GCS_BUCKET_NAME, self.creds_path
                )

        return build("gmail", "v1", credentials=creds)

    def fetch_emails(self, query: str, max_results: int = None) -> List[Dict]:
        """
        Fetches emails based on a specific query.

        Args:
            query: The query to fetch the emails.
            max_results: The maximum number of emails to fetch.

        Returns:
            A list of messages.
        """
        all_emails = []
        request = self.service.users().messages().list(userId="me", q=query)

        while request is not None:
            response = request.execute()
            logger.debug(f"Response: {response}")
            all_emails.extend(response.get("messages", []))
            logger.debug(f"All emails: {all_emails}")

            # If max_results is specified and we've fetched enough emails, break the loop.
            if max_results is not None and len(all_emails) >= max_results:
                logger.info("Breaking due to max_results")
                break

            request = self.service.users().messages().list_next(request, response)
            logger.debug(f"Next request: {request}")

        return all_emails[:max_results]

    def get_email_data(self, email_id: str) -> Dict:
        """
        Gets data of a specific email.

        Args:
            email_id: The email ID.

        Returns:
            The email data.
        """

        return self.service.users().messages().get(userId="me", id=email_id).execute()

    def fetch_labels(self) -> List[Dict]:
        """
        Fetches all labels from the user's Gmail account.

        Returns:
            The list of labels.
        """

        return (
            self.service.users().labels().list(userId="me").execute().get("labels", [])
        )

    def get_label_id(self, label_name: str) -> str:
        """
        Translates a label name to its corresponding ID.

        Args:
            label_name: The label name.

        Returns:
            The label ID.

        Raises:
            ValueError: If no label is found with the given name.
        """

        labels = self.fetch_labels()
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        logger.error(f"No label found with the name {label_name}")
        raise ValueError(f"No label found with the name {label_name}")

    def get_body(self, email: Dict) -> str:
        """
        Gets the body of the email.

        Args:
            email: The email data.

        Returns:
            The email body.
        """

        parts = email["payload"].get("parts")
        data = parts[0]["body"]["data"] if parts else email["payload"]["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8")

    def get_articles_from_emails(
        self, emails: List[Dict], content_parser: Callable
    ) -> List:
        """
        Extracts articles from a list of emails using a specific content parser.

        Args:
            emails: The list of emails.
            content_parser: The function to parse the content.

        Returns:
            A list of articles.
        """

        articles = []
        for email in emails:
            email_data = self.get_email_data(email["id"])
            content = self.get_body(email_data)
            articles += content_parser(content)
        return articles

    def trash_emails(self, email_ids: List[str]) -> List[Dict]:
        """
        Moves emails to the trash in the user's Gmail account.

        Args:
            email_ids: A list of email IDs.

        Returns:
            A list of responses, each containing the result for an individual email.
        """
        responses = []

        for email_id in email_ids:
            result = (
                self.service.users()
                .messages()
                .trash(userId="me", id=email_id)
                .execute()
            )
            responses.append(result)
            logger.info(f"Moved email {email_id} to trash.")

        return responses
