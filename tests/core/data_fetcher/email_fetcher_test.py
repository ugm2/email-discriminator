from unittest.mock import MagicMock, call, mock_open, patch

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
    mock_service = mock_get_service.return_value
    mock_list_method = mock_service.users.return_value.messages.return_value.list
    mock_list_next_method = (
        mock_service.users.return_value.messages.return_value.list_next
    )

    # Define a generator function that will be used as the side_effect for execute.
    def execute_side_effect():
        yield {"messages": ["Email1", "Email2", "Email3"]}  # First page
        yield {"messages": ["Email4", "Email5"]}  # Second page
        while True:  # Ensure we don't raise StopIteration
            yield {"messages": []}

    # Set the side_effect of execute to the generator function.
    mock_list_method.return_value.execute.side_effect = execute_side_effect()

    # Simulate the list_next method.
    mock_list_next_method.side_effect = lambda x, y: (
        mock_service.users.return_value.messages.return_value.list.return_value  # Second page
        if y.get("messages") == ["Email1", "Email2", "Email3"]
        else None  # No more pages
    )

    fetcher = EmailFetcher()

    # Test without specifying max_results.
    emails = fetcher.fetch_emails("label:TLDRs")
    assert emails == ["Email1", "Email2", "Email3", "Email4", "Email5"]
    assert mock_list_method.call_args_list == [call(userId="me", q="label:TLDRs")]

    # Reset the mocks.
    mock_list_method.reset_mock()
    mock_list_next_method.reset_mock()

    # Reset the side effect.
    mock_list_method.return_value.execute.side_effect = execute_side_effect()

    # Test with max_results.
    emails = fetcher.fetch_emails("label:TLDRs", max_results=3)
    assert emails == ["Email1", "Email2", "Email3"]
    assert mock_list_method.call_args_list == [call(userId="me", q="label:TLDRs")]

    # Reset the mocks.
    mock_list_method.reset_mock()
    mock_list_next_method.reset_mock()

    # TODO: Test with max_results greater than the total number of emails.
    # emails = fetcher.fetch_emails("label:TLDRs", max_results=10)
    # print(f"Emails with max_results=10: {emails}")
    # assert emails == ["Email1", "Email2", "Email3", "Email4", "Email5"]
    # assert mock_list_method.call_args_list == [call(userId="me", q="label:TLDRs")]


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


@patch.object(EmailFetcher, "get_service", return_value=MagicMock())
def test_delete_emails(mock_get_service):
    mock_service = mock_get_service.return_value
    mock_batch_delete_method = (
        mock_service.users.return_value.messages.return_value.batchDelete
    )
    mock_batch_delete_method.return_value.execute.return_value = "Result"

    fetcher = EmailFetcher()

    email_ids = ["id1", "id2", "id3"]
    result = fetcher.delete_emails(email_ids)

    assert result == "Result"

    mock_batch_delete_method.assert_called_once_with(
        userId="me", body={"ids": email_ids}
    )
