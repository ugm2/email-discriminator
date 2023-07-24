import pandas as pd

from email_discriminator.core.model import DataProcessor, TextSelector


def test_text_selector():
    selector = TextSelector(key="a")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Testing transform method
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

    # Testing fit_transform method
    transformed = data_processor.fit_transform(df)
    assert (
        transformed.shape[1] == 3
    ), "DataProcessor fit_transform output shape is incorrect."
