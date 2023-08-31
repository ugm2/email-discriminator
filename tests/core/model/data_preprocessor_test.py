from unittest.mock import patch

import pandas as pd
import pytest

from email_discriminator.core.model import DataProcessor, TextSelector


def test_text_selector():
    selector = TextSelector(key="a")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    transformed = selector.transform(df)
    assert transformed.tolist() == [1, 2, 3], "TextSelector transform is incorrect."


def test_data_processor():
    data_processor = DataProcessor()
    df = pd.DataFrame(
        {
            "article": ["This is a test", "Another test", "Final test"],
            "section": ["cat", "dog", "bird"],
        }
    )
    transformed = data_processor.fit_transform(df)
    assert (
        transformed.shape[1] >= 3
    ), "DataProcessor fit_transform output shape is incorrect."


def test_data_processor_empty_dataframe():
    data_processor = DataProcessor()
    df = pd.DataFrame()
    with pytest.raises(ValueError) as e:
        data_processor.fit_transform(df)
    assert (
        str(e.value) == "Input DataFrame is empty!"
    ), "Empty DataFrame error handling is incorrect."


def test_text_selector_invalid_key():
    selector = TextSelector(key="c")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError) as e:
        selector.transform(df)
    assert "Column `c` not found! Available columns: a, b" in str(
        e.value
    ), "Invalid key error handling is incorrect."


def test_data_processor_exceptions():
    data_processor = DataProcessor()
    df = pd.DataFrame(
        {"article": ["This is a test", "Another test"], "section": ["cat", "dog"]}
    )

    # Test fit exception
    with patch(
        "email_discriminator.core.model.data_processor.FeatureUnion.fit"
    ) as mock_feature_fit:
        # Making the mock throw an exception when called
        mock_feature_fit.side_effect = Exception("Error fitting data")

        with pytest.raises(Exception) as e:
            data_processor.fit(df)

        assert "Error fitting data" in str(
            e.value
        ), "Expected exception message not found."

    # Test transform exception
    with patch(
        "email_discriminator.core.model.data_processor.FeatureUnion.transform"
    ) as mock_feature_fit:
        # Making the mock throw an exception when called
        mock_feature_fit.side_effect = Exception("Error transforming data")

        with pytest.raises(Exception) as e:
            data_processor.transform(df)

        assert "Error transforming data" in str(
            e.value
        ), "Expected exception message not found."


def test_data_processor_individual_component():
    data_processor = DataProcessor()
    df = pd.DataFrame({"article": ["This is a test"], "section": ["cat"]})

    data_processor.fit(df)

    # Test 'article' pipeline
    text_pipeline = data_processor.text
    text_transformed = text_pipeline.transform(df)
    assert (
        text_transformed.shape[1] >= 1
    ), "'article' pipeline output shape is incorrect."  # Revise this as needed

    # Test 'section' pipeline
    section_pipeline = data_processor.section
    section_transformed = section_pipeline.transform(df)
    assert (
        section_transformed.shape[0] == 1
    ), "'section' pipeline output shape is incorrect."
