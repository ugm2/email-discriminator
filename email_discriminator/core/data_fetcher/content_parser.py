import re
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from rich.logging import RichHandler
import logging
import os

# Logger configuration
LOGGER_LEVEL = os.environ.get("LOGGER_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=LOGGER_LEVEL, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("TLDRContentParser")


class ContentParserInterface(ABC):
    """
    Abstract base class for content parsers.
    """

    @abstractmethod
    def parse_content(self, content: str) -> List[Dict[str, Union[str, List[str]]]]:
        pass


class TLDRContentParser(ContentParserInterface):
    """
    Class for parsing content.
    """

    def __init__(self):
        """
        Initialize TLDRContentParser with default sections.
        """
        self.sections = {
            "BIG TECH & STARTUPS": None,
            "SCIENCE & FUTURISTIC TECHNOLOGY": None,
            "PROGRAMMING, DESIGN & DATA SCIENCE": None,
            "MISCELLANEOUS": None,
            "QUICK LINKS": None,
        }
        logger.info(
            f"TLDRContentParser initialized with sections: {self.sections.keys()}"
        )

    def parse_content(self, content: str) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Extract TLDR articles from content.

        Args:
            content: The email content.

        Returns:
            A list of dictionaries each containing the section and the articles.
        """
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
        logger.info(f"Parsed {len(tldr_articles)} articles from the content")
        return tldr_articles

    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from TLDR content.

        Args:
            content: The email content.

        Returns:
            A dictionary with the section titles as keys and the contents as values.
        """
        pattern = rf"[\u263a-\U0001f645]*\s*({ '|'.join(self.sections.keys()) })\s*[\r\n]+(.*?)(?=[\u263a-\U0001f645]*\s*({ '|'.join(self.sections.keys()) })\s*[\r\n]+|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for section, section_content, _ in matches:
            self.sections[section] = section_content.strip()
        logger.info(f"Extracted {len(self.sections)} sections from the content")
        return self.sections

    def extract_articles(self, section_content: str) -> List[str]:
        """
        Extract articles from a section.

        Args:
            section_content: The content of the section.

        Returns:
            A list of articles.
        """
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

        logger.info(f"Extracted {len(articles)} articles from the section content")
        return articles


if __name__ == "__main__":
    from email_discriminator.core.data_fetcher.email_fetcher import EmailFetcher

    fetcher = EmailFetcher()
    parser = TLDRContentParser()

    relevant_emails = fetcher.fetch_emails("label:TLDRs")

    relevant_articles = fetcher.get_articles_from_emails(
        relevant_emails, parser.parse_content
    )
    print(relevant_articles[:2])
