from unittest.mock import MagicMock, mock_open, patch

import pytest

from email_discriminator.core.data_fetcher import EmailFetcher


@patch("builtins.open", new_callable=mock_open, read_data="token.pickle")
@patch("email_discriminator.core.data_fetcher.email_fetcher.build")
@patch(
    "email_discriminator.core.data_fetcher.email_fetcher.InstalledAppFlow.from_client_secrets_file"
)
@patch("email_discriminator.core.data_fetcher.email_fetcher.pickle.load")
@patch("email_discriminator.core.data_fetcher.email_fetcher.os.path.exists")
def test_get_service(mock_exists, mock_load, mock_flow, mock_build, mock_open):
    mock_exists.return_value = True
    mock_load.return_value = MagicMock(valid=True)
    mock_build.return_value = "Service"
    mock_open.return_value.__enter__.return_value = mock_open.return_value

    fetcher = EmailFetcher()
    assert fetcher.service == "Service"


@patch.object(EmailFetcher, "get_service", return_value=MagicMock())
def test_fetch_emails(mock_get_service):
    mock_get_service.return_value.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": "Emails"
    }
    fetcher = EmailFetcher()
    assert fetcher.fetch_emails("label:TLDRs") == "Emails"


@patch.object(EmailFetcher, "get_service", return_value=MagicMock())
def test_get_email_data(mock_get_service):
    mock_get_service.return_value.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "emailData": "Data"
    }
    fetcher = EmailFetcher()
    assert fetcher.get_email_data("id") == {"emailData": "Data"}


@patch.object(EmailFetcher, "get_service", return_value=MagicMock())
def test_fetch_labels(mock_get_service):
    mock_get_service.return_value.users.return_value.labels.return_value.list.return_value.execute.return_value = {
        "labels": "Labels"
    }
    fetcher = EmailFetcher()
    assert fetcher.fetch_labels() == "Labels"


@patch.object(
    EmailFetcher, "fetch_labels", return_value=[{"name": "Label", "id": "ID"}]
)
@patch.object(EmailFetcher, "get_service", return_value=None)
def test_get_label_id(mock_get_service, mock_fetch_labels):
    fetcher = EmailFetcher()
    assert fetcher.get_label_id("Label") == "ID"
    with pytest.raises(ValueError):
        fetcher.get_label_id("No Label")


@patch.object(EmailFetcher, "get_service", return_value=MagicMock())
def test_get_body(mock_get_service):
    mock_get_service.return_value.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "payload": {"body": {"data": "aGVsbG8="}}
    }  # 'aGVsbG8=' is 'hello' in base64
    fetcher = EmailFetcher()
    assert (
        fetcher.get_body({"id": "id", "payload": {"body": {"data": "aGVsbG8="}}})
        == "hello"
    )


@patch.object(
    EmailFetcher,
    "get_email_data",
    return_value={"payload": {"body": {"data": "aGVsbG8gd29ybGQ="}}},
)  # 'aGVsbG8gd29ybGQ=' is 'hello world' in base64
@patch.object(EmailFetcher, "get_body", return_value="hello world")
@patch.object(EmailFetcher, "get_service", return_value=None)
def test_get_articles_from_emails(mock_get_service, mock_get_body, mock_get_email_data):
    emails = [{"id": "1"}, {"id": "2"}]
    content_parser = lambda content: content.split()

    fetcher = EmailFetcher()
    articles = fetcher.get_articles_from_emails(emails, content_parser)

    assert articles == ["hello", "world", "hello", "world"]

    mock_get_email_data.assert_called()
    mock_get_body.assert_called()
