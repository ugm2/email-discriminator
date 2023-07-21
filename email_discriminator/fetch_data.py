import os
import pickle
import re
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import base64
from rich import print
from abc import ABC, abstractmethod
import pandas as pd


class EmailFetcher:
    def __init__(
        self, creds_path="token.pickle", client_secret_path="client_secret.json"
    ):
        self.creds_path = creds_path
        self.client_secret_path = client_secret_path
        self.service = self.get_service()

    def get_service(self):
        """Authenticate and return a service object for the Gmail API."""
        creds = None
        if os.path.exists(self.creds_path):
            with open(self.creds_path, "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_path,
                    ["https://www.googleapis.com/auth/gmail.modify"],
                )
                creds = flow.run_local_server(port=0)
            with open(self.creds_path, "wb") as token:
                pickle.dump(creds, token)
        return build("gmail", "v1", credentials=creds)

    def fetch_emails(self, query):
        """Fetch emails based on a specific query."""
        return (
            self.service.users()
            .messages()
            .list(userId="me", q=query)
            .execute()
            .get("messages", [])
        )

    def get_email_data(self, email_id):
        """Get data of a specific email."""
        return self.service.users().messages().get(userId="me", id=email_id).execute()

    def get_articles_from_emails(self, emails, content_parser):
        """Extract articles from a list of emails using a specific content parser."""
        articles = []
        for email in emails:
            email_data = self.get_email_data(email["id"])
            content = self.get_body(email_data)
            articles += content_parser(content)
        return articles

    def fetch_labels(self):
        """Fetch all labels from the user's Gmail account."""
        return (
            self.service.users().labels().list(userId="me").execute().get("labels", [])
        )

    def get_label_id(self, label_name):
        """Translate a label name to its corresponding ID."""
        labels = self.fetch_labels()
        for label in labels:
            if label["name"] == label_name:
                return label["id"]
        raise ValueError(f"No label found with the name {label_name}")

    def get_body(self, email):
        """Get the body of the email."""
        if "parts" in email["payload"]:
            return base64.urlsafe_b64decode(
                email["payload"]["parts"][0]["body"]["data"]
            ).decode("utf-8")
        else:
            return base64.urlsafe_b64decode(email["payload"]["body"]["data"]).decode(
                "utf-8"
            )


class ContentParser(ABC):
    """Abstract base class for content parsers."""

    @abstractmethod
    def parse_content(self, content):
        pass


class TLDRContentParser(ContentParser):
    def __init__(self):
        self.sections = {
            "BIG TECH & STARTUPS": None,
            "SCIENCE & FUTURISTIC TECHNOLOGY": None,
            "PROGRAMMING, DESIGN & DATA SCIENCE": None,
            "MISCELLANEOUS": None,
            "QUICK LINKS": None,
        }

    def parse_content(self, content):
        """Extract TLDR articles from content."""
        tldr_articles = []
        sections_content = self.extract_sections(content)

        for section, section_content in sections_content.items():
            articles = self.extract_articles(section_content)
            for article in articles:
                tldr_articles.append(
                    {
                        "section": section,
                        "article": article,
                    }
                )
        return tldr_articles

    def extract_sections(self, content):
        """Extract sections from TLDR content."""
        pattern = rf"[\u263a-\U0001f645]*\s*({ '|'.join(self.sections.keys()) })\s*[\r\n]+(.*?)(?=[\u263a-\U0001f645]*\s*({ '|'.join(self.sections.keys()) })\s*[\r\n]+|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for section, section_content, _ in matches:
            self.sections[section] = section_content.strip()
        return self.sections

    def extract_articles(self, section_content):
        """Extract articles from a section."""
        if section_content is None:
            return []

        # Split the section content into articles
        articles = section_content.split("\r\n\r\n")

        # Combine the title and content into the same chunk
        articles = [
            "\r\n\r\n".join(articles[i : i + 2]) for i in range(0, len(articles), 2)
        ]
        # Filter out strings that don't appear to be articles
        articles = [
            article
            for article in articles
            if re.search(r"(MINUTE\s*READ)|(GITHUB\s*REPO)", article, re.IGNORECASE)
        ]

        return articles


if __name__ == "__main__":
    fetcher = EmailFetcher()
    tldr_parser = TLDRContentParser()

    # Fetch emails from the relevant and irrelevant labels
    relevant_emails = fetcher.fetch_emails("label:TLDRs")
    irrelevant_emails = fetcher.fetch_emails(
        f"from:dan@tldrnewsletter.com is:read -label:TLDRs"
    )
    # Extract the TLDR articles from the emails
    relevant_articles = fetcher.get_articles_from_emails(
        relevant_emails, tldr_parser.parse_content
    )
    irrelevant_articles = fetcher.get_articles_from_emails(
        irrelevant_emails, tldr_parser.parse_content
    )

    print("TLDR articles:")
    print(relevant_articles[:5])
    print("\n\n")
    print("Irrelevant articles:")
    print(irrelevant_articles[:5])

    # Convert the lists of relevant and irrelevant articles into pandas DataFrames
    relevant_df = pd.DataFrame(relevant_articles)
    relevant_df["is_relevant"] = 1

    irrelevant_df = pd.DataFrame(irrelevant_articles)
    irrelevant_df["is_relevant"] = 0

    # Concatenate the two DataFrames into a single DataFrame
    articles_df = pd.concat([relevant_df, irrelevant_df], ignore_index=True)

    print(articles_df.head())
    print(articles_df["is_relevant"].value_counts())
