import os
import pickle
import re
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import base64
from rich import print


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

    def fetch_emails(self, label_name):
        """Fetch emails from a specific Gmail label."""
        label_id = self.get_label_id(label_name)
        return (
            self.service.users()
            .messages()
            .list(userId="me", labelIds=[label_id])
            .execute()
            .get("messages", [])
        )

    def fetch_email_data(self, email_id):
        """Fetch the data of a specific email."""
        return self.service.users().messages().get(userId="me", id=email_id).execute()

    def fetch_tldr_articles(self, emails):
        """Extract TLDR articles from a list of emails."""
        tldr_articles = []
        for email in emails:
            email_data = self.fetch_email_data(email["id"])
            content = self.get_body(email_data)

            sections = self.extract_sections(content)

            for section, section_content in sections.items():
                articles = self.extract_articles(section_content)
                for article in articles:
                    tldr_articles.append(
                        {
                            "email_id": email["id"],
                            "section": section,
                            "article": article,
                        }
                    )
        return tldr_articles

    def get_articles_by_section(self, content):
        """Split content into sections and extract articles from each section."""
        sections = self.extract_sections(content)
        articles_by_section = {}
        for i, section in enumerate(sections):
            articles = self.extract_articles(section)
            articles_by_section[i] = articles
        return articles_by_section

    def extract_sections(self, content):
        """Extract sections from TLDR content."""
        sections = {
            "BIG TECH & STARTUPS": None,
            "SCIENCE & FUTURISTIC TECHNOLOGY": None,
            "PROGRAMMING, DESIGN & DATA SCIENCE": None,
            "MISCELLANEOUS": None,
            "QUICK LINKS": None,
        }
        pattern = rf"[\u263a-\U0001f645]*\s*({ '|'.join(sections.keys()) })\s*[\r\n]+(.*?)(?=[\u263a-\U0001f645]*\s*({ '|'.join(sections.keys()) })\s*[\r\n]+|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for section, section_content, _ in matches:
            sections[section] = section_content.strip()
        return sections

    def extract_articles(self, section_content):
        """Extract articles from a section."""
        if section_content is None:
            return []

        # TODO: Improve pattern
        # Define the pattern to identify each article
        pattern = r"\(\d(?:.+| \n)MINUTE(?:.+| \n)READ\)"

        # Use findall to capture all instances that match the pattern
        articles = re.findall(pattern, section_content, re.DOTALL)
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


if __name__ == "__main__":
    fetcher = EmailFetcher()

    # Fetch emails from the relevant and irrelevant labels
    relevant_emails = fetcher.fetch_emails("TLDRs")
    # irrelevant_emails = fetcher.fetch_emails(
    #     "Archived"
    # )  # replace "Archived" with the actual name of your label

    # Extract the TLDR articles from the emails
    relevant_articles = fetcher.fetch_tldr_articles(relevant_emails)
    # irrelevant_articles = fetcher.fetch_tldr_articles(irrelevant_emails)

    print("TLDR articles:")
    print(relevant_articles[:2])
    # print("\n\n")
    # print("Irrelevant articles:")
    # print(irrelevant_articles[:5])
