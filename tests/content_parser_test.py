import pytest

from email_discriminator.core.data_fetcher.content_parser import TLDRContentParser


@pytest.fixture
def tldr_content_parser():
    return TLDRContentParser()


def test_extract_sections(tldr_content_parser):
    content = "🚀 BIG TECH & STARTUPS\n\nArticle 1\n\nArticle 2\n\n😎 PROGRAMMING, DESIGN & DATA SCIENCE\n\nArticle 3\n\nArticle 4"
    sections = tldr_content_parser.extract_sections(content)
    assert sections["BIG TECH & STARTUPS"] == "Article 1\n\nArticle 2"
    assert sections["PROGRAMMING, DESIGN & DATA SCIENCE"] == "Article 3\n\nArticle 4"


def test_extract_articles(tldr_content_parser):
    section_content = (
        "Article 1\r\n\r\n3 MINUTE READ\r\n\r\nArticle 2\r\n\r\nGITHUB REPO"
    )
    articles = tldr_content_parser.extract_articles(section_content)
    assert len(articles) == 2
    assert articles[0] == "Article 1\r\n\r\n3 MINUTE READ"
    assert articles[1] == "Article 2\r\n\r\nGITHUB REPO"


def test_parse_content(tldr_content_parser):
    content = "🚀 BIG TECH & STARTUPS\r\n\r\nArticle 1\r\n\r\n3 MINUTE READ\r\n\r\nArticle 2\r\n\r\nGITHUB REPO"
    articles = tldr_content_parser.parse_content(content)
    assert len(articles) == 2
    assert articles[0] == {
        "section": "BIG TECH & STARTUPS",
        "article": "Article 1\r\n\r\n3 MINUTE READ",
    }
    assert articles[1] == {
        "section": "BIG TECH & STARTUPS",
        "article": "Article 2\r\n\r\nGITHUB REPO",
    }
