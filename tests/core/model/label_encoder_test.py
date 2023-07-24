import numpy as np

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
