import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from email_discriminator.core.model import DataProcessor, Model, Pipeline


def test_pipeline():
    data_processor = DataProcessor()
    model = Model()
    pipeline = Pipeline(data_processor, model)

    X = pd.DataFrame(
        {
            "article": ["This is a test", "Another test", "Final test"],
            "section": ["cat", "dog", "bird"],
        }
    )
    y = np.array([0, 1, 1])

    # Testing fit and predict methods
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == 3, "Pipeline predict output length is incorrect."

    # Testing error raising when not fitted
    new_pipeline = Pipeline(data_processor, model)
    with pytest.raises(NotFittedError):
        new_pipeline.predict(X)


def test_predict_proba():
    data_processor = DataProcessor()
    model = Model()
    pipeline = Pipeline(data_processor, model)

    X = pd.DataFrame(
        {
            "article": ["This is a test", "Another test", "Final test"],
            "section": ["cat", "dog", "bird"],
        }
    )
    y = np.array([0, 1, 1])

    # Testing fit and predict_proba methods
    pipeline.fit(X, y)
    prob_preds = pipeline.predict_proba(X)

    assert len(prob_preds) == 3, "Pipeline predict_proba output length is incorrect."
    assert all(
        0 <= prob_pred[1] <= 1 for prob_pred in prob_preds
    ), "Pipeline predict_proba prediction not in range [0, 1]"


def test_save_and_load():
    # Initialize objects
    data_processor = DataProcessor()
    model = Model()
    pipeline = Pipeline(data_processor, model)

    X = pd.DataFrame(
        {
            "article": ["This is a test", "Another test", "Final test"],
            "section": ["cat", "dog", "bird"],
        }
    )
    y = np.array([0, 1, 1])

    # Fit the model
    pipeline.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "test_model.joblib")

        # Save the model
        pipeline.save(model_path)

        # Assert model file was created
        assert os.path.isfile(model_path), "Model was not saved."

        # Load the model
        loaded_pipeline = Pipeline.load(model_path)

        # Assert the loaded model works as expected
        preds = loaded_pipeline.predict(X)
        assert len(preds) == 3, "Loaded model predict output length is incorrect."
