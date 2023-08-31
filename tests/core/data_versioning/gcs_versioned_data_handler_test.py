import json
import pickle
from unittest.mock import MagicMock, Mock, patch

import pytest

from email_discriminator.core.data_versioning import GCSVersionedDataHandler

# Mocked objects for our tests
mock_bucket = MagicMock()
mock_blob = MagicMock()


# Fixture for a fresh instance of the handler
@pytest.fixture
def handler():
    with patch(
        "email_discriminator.core.data_versioning.gcs_versioned_data_handler.storage.Client"
    ) as MockClient:
        MockClient.return_value.get_bucket.return_value = mock_bucket
        MockClient.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        yield GCSVersionedDataHandler("test-bucket")
        mock_bucket.reset_mock()


def test_enable_versioning(handler):
    handler.enable_versioning()
    assert mock_bucket.versioning_enabled == True
    mock_bucket.patch.assert_called_once()


def test_disable_versioning(handler):
    handler.disable_versioning()
    assert mock_bucket.versioning_enabled == False
    mock_bucket.patch.assert_called_once()


def test_upload_string(handler):
    test_str = "test"
    test_path = "path/to/test.txt"
    handler.upload_string(test_str, test_path)
    mock_blob.upload_from_string.assert_called_with(test_str)


def test_download_string_without_version(handler):
    test_path = "path/to/test.txt"
    handler.download_string(test_path)
    mock_bucket.blob.assert_called_with(test_path, generation=None)
    assert mock_blob.download_as_text.called


def test_download_string_with_version(handler):
    version = 1234
    test_path = "path/to/test.txt"
    handler.download_string(test_path, version)
    mock_bucket.blob.assert_called_with(test_path, generation=version)
    assert mock_blob.download_as_text.called


def test_move_file_string(handler):
    src_path = "path/to/source.txt"
    dest_path = "path/to/dest.txt"
    handler.move_file_string(src_path, dest_path)
    mock_blob.move_to.assert_called_once()


def test_upload_original_data(handler):
    handler.upload_original_data("path/on/disk.csv")
    mock_blob.upload_from_string.assert_called_with("path/on/disk.csv")


def test_download_original_data(handler):
    handler.download_original_data()
    mock_bucket.blob.assert_called_with(
        "data/original_data/tldr_articles.csv", generation=None
    )
    assert mock_blob.download_as_text.called


def test_upload_unlabelled_data(handler):
    csv_string = "test_csv"
    data_hash = "test_hash"
    handler.upload_unlabelled_data(csv_string, data_hash)
    mock_blob.upload_from_string.assert_called_with(csv_string)


def test_download_unlabelled_data(handler):
    data_hash = "test_hash"
    handler.download_unlabelled_data(data_hash)
    mock_bucket.blob.assert_called_with(
        f"data/unlabelled_data/tldr_articles_{data_hash}.csv", generation=None
    )
    assert mock_blob.download_as_text.called


def test_upload_predicted_data(handler):
    csv_string = "test_csv"
    data_hash = "test_hash"
    folder = "test_folder"
    handler.upload_predicted_data(csv_string, data_hash, folder)
    mock_blob.upload_from_string.assert_called_with(csv_string)


def test_download_predicted_data(handler):
    data_hash = "test_hash"
    handler.download_predicted_data(data_hash)
    mock_bucket.blob.assert_called_with(
        f"data/predicted_data/new/tldr_articles_{data_hash}.csv", generation=None
    )
    assert mock_blob.download_as_text.called


def test_move_predicted_data_to_old(handler):
    data_hash = "test_hash"
    handler.move_predicted_data_to_old(data_hash)
    mock_blob.move_to.assert_called_once()


def test_upload_training_data(handler):
    csv_string = "test_csv"
    data_hash = "test_hash"
    handler.upload_training_data(csv_string, data_hash)
    mock_blob.upload_from_string.assert_called_with(csv_string)


def test_download_training_data(handler):
    data_hash = "test_hash"
    handler.download_training_data(data_hash)
    mock_bucket.blob.assert_called_with(
        f"data/training_data/tldr_articles_{data_hash}.csv", generation=None
    )
    assert mock_blob.download_as_text.called


def test_download_all_training_data(handler):
    blobs = [
        MagicMock(name=f"data/training_data/tldr_articles_{i}.csv") for i in range(5)
    ]
    mock_bucket.list_blobs.return_value = blobs
    result = handler.download_all_training_data()
    assert len(result) == 5


def test_download_new_predicted_data(handler):
    blobs = [
        MagicMock(name=f"data/predicted_data/new/tldr_articles_{i}.csv")
        for i in range(5)
    ]
    mock_bucket.list_blobs.return_value = blobs
    result = handler.download_new_predicted_data()
    assert len(result) == 5


def test_delete_predicted_file(handler):
    data_hash = "test_hash"
    handler.delete_predicted_file(data_hash)
    mock_blob.delete.assert_called_once()


def test_read_token_from_gcs(handler):
    token = "token"
    mock_token = pickle.dumps(token)
    mock_blob.download_as_bytes.return_value = mock_token
    result = handler.read_token_from_gcs("test-bucket", "blob-name")
    assert result == token


def test_write_token_to_gcs(handler):
    test_creds = "credentials"
    handler.write_token_to_gcs(test_creds, "test-bucket", "blob-name")
    mock_blob.upload_from_string.assert_called_with(pickle.dumps(test_creds))


def test_read_client_secret_from_gcs(handler):
    client_secrets_json = {"test": "test"}
    mock_blob.download_as_text.return_value = json.dumps(client_secrets_json)
    result = handler.read_client_secrets_from_gcs("test-bucket", "blob-name")
    assert result == client_secrets_json
