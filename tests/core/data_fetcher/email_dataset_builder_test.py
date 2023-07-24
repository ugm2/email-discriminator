from unittest.mock import Mock, patch

import pandas as pd
import pytest

from email_discriminator.core.data_fetcher import EmailDatasetBuilder


@pytest.fixture
def fetcher_mock():
    return Mock()


@pytest.fixture
def parser_mock():
    return Mock()


@pytest.fixture
def email_dataset_builder(fetcher_mock, parser_mock):
    return EmailDatasetBuilder(fetcher_mock, parser_mock)


def test_create_training_dataframe(email_dataset_builder, fetcher_mock, parser_mock):
    # Mock the behavior of fetcher and parser
    fetcher_mock.fetch_emails.return_value = ["email1", "email2"]
    fetcher_mock.get_articles_from_emails.return_value = ["article1", "article2"]
    parser_mock.parse_content.return_value = ["parsed_article1", "parsed_article2"]

    df = email_dataset_builder.create_training_dataframe(
        "relevant_query", "irrelevant_query"
    )

    # Verify the returned dataframe
    assert isinstance(df, pd.DataFrame)
    assert df["is_relevant"].value_counts().tolist() == [2, 2]


def test_create_predict_dataframe(email_dataset_builder, fetcher_mock, parser_mock):
    # Mock the behavior of fetcher and parser
    fetcher_mock.fetch_emails.return_value = ["email1", "email2"]
    fetcher_mock.get_articles_from_emails.return_value = ["article1", "article2"]
    parser_mock.parse_content.return_value = ["parsed_article1", "parsed_article2"]

    df = email_dataset_builder.create_predict_dataframe("query")

    # Verify the returned dataframe
    assert isinstance(df, pd.DataFrame)
    assert "is_relevant" not in df.columns


@patch("email_discriminator.core.data_fetcher.email_dataset_builder.logger")
def test_create_training_dataframe_exception_on_fetch(
    mock_logger, email_dataset_builder, fetcher_mock, parser_mock
):
    # Simulate an exception when calling fetch_emails
    fetcher_mock.fetch_emails.side_effect = Exception("Fetch error")

    with pytest.raises(Exception, match="Fetch error"):
        email_dataset_builder.create_training_dataframe(
            "relevant_query", "irrelevant_query"
        )

    # Assert that the error log was called
    mock_logger.error.assert_called_with("Failed to fetch emails due to Fetch error")


@patch("email_discriminator.core.data_fetcher.email_dataset_builder.logger")
def test_create_training_dataframe_exception_on_get_articles(
    mock_logger, email_dataset_builder, fetcher_mock, parser_mock
):
    # Simulate an exception when calling get_articles_from_emails
    fetcher_mock.fetch_emails.return_value = ["email1", "email2"]
    fetcher_mock.get_articles_from_emails.side_effect = Exception("Articles error")

    with pytest.raises(Exception, match="Articles error"):
        email_dataset_builder.create_training_dataframe(
            "relevant_query", "irrelevant_query"
        )

    # Assert that the error log was called
    mock_logger.error.assert_called_with(
        "Failed to get articles from emails due to Articles error"
    )


@patch("email_discriminator.core.data_fetcher.email_dataset_builder.logger")
def test_create_predict_dataframe_exception_on_fetch(
    mock_logger, email_dataset_builder, fetcher_mock, parser_mock
):
    # Simulate an exception when calling fetch_emails
    fetcher_mock.fetch_emails.side_effect = Exception("Fetch error")

    with pytest.raises(Exception, match="Fetch error"):
        email_dataset_builder.create_predict_dataframe("query")

    # Assert that the error log was called
    mock_logger.error.assert_called_with("Failed to fetch emails due to Fetch error")


@patch("email_discriminator.core.data_fetcher.email_dataset_builder.logger")
def test_create_predict_dataframe_exception_on_get_articles(
    mock_logger, email_dataset_builder, fetcher_mock, parser_mock
):
    # Simulate an exception when calling get_articles_from_emails
    fetcher_mock.fetch_emails.return_value = ["email1", "email2"]
    fetcher_mock.get_articles_from_emails.side_effect = Exception("Articles error")

    with pytest.raises(Exception, match="Articles error"):
        email_dataset_builder.create_predict_dataframe("query")

    # Assert that the error log was called
    mock_logger.error.assert_called_with(
        "Failed to get articles from emails due to Articles error"
    )
