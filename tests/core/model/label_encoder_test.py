import numpy as np
from sklearn.pipeline import Pipeline

from email_discriminator.core.model import CustomLabelEncoder


def test_custom_label_encoder():
    encoder = CustomLabelEncoder()
    data = np.array(["cat", "dog", "bird"])

    # Testing fit and transform methods
    encoder.fit(data)
    transformed = encoder.transform(data)

    # Check that all labels are in the correct range
    assert all(
        0 <= label < len(np.unique(data)) for label in transformed
    ), "CustomLabelEncoder transform is incorrect."

    # Check that transformed labels match the original data when inverse transformed
    inverse_transformed = encoder.encoder.inverse_transform(transformed.ravel())
    assert np.array_equal(
        inverse_transformed, data
    ), "CustomLabelEncoder inverse transform doesn't match original data."


def test_custom_label_encoder_empty_array():
    encoder = CustomLabelEncoder()
    data = np.array([])

    encoder.fit(data)
    transformed = encoder.transform(data)
    assert len(transformed) == 0, "Transformed data of an empty array should be empty."


def test_custom_label_encoder_numeric_labels():
    encoder = CustomLabelEncoder()
    data = np.array([1, 2, 3])

    encoder.fit(data)
    transformed = encoder.transform(data)

    assert all(
        0 <= label < len(np.unique(data)) for label in transformed
    ), "CustomLabelEncoder transform is incorrect for numeric labels."


def test_custom_label_encoder_single_class():
    encoder = CustomLabelEncoder()
    data = np.array(["cat", "cat", "cat"])

    encoder.fit(data)
    transformed = encoder.transform(data)

    assert all(
        label == 0 for label in transformed
    ), "CustomLabelEncoder should assign 0 for single-class labels."


import pytest


def test_custom_label_encoder_exceptions():
    encoder = CustomLabelEncoder()
    with pytest.raises(Exception):
        encoder.transform(np.array(["unknown_label"]))


def test_custom_label_encoder_pipeline():
    pipeline = Pipeline([("encoder", CustomLabelEncoder())])
    data = np.array(["cat", "dog", "bird"])
    pipeline.fit(data)
    transformed = pipeline.transform(data)

    assert all(
        0 <= label < len(np.unique(data)) for label in transformed
    ), "CustomLabelEncoder transform is incorrect in a pipeline."


def test_custom_label_encoder_fit_error():
    encoder = CustomLabelEncoder()
    invalid_data = np.array(
        [[1, 2], [3, 4]]
    )  # Two-dimensional array which should not be supported by LabelEncoder

    with pytest.raises(Exception) as e:
        encoder.fit(invalid_data)

    assert (
        str(e.value) == "y should be a 1d array, got an array of shape (2, 2) instead."
    ), "CustomLabelEncoder does not raise an exception or has a different error message when fitting invalid data."
