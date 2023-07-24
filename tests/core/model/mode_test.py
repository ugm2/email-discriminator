import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from email_discriminator.core.model import Model


def test_model():
    model = Model()
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    # Testing fit and predict methods
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 100, "Model predict output length is incorrect."


def test_fit_predict():
    model = Model()
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    # Testing fit and predict methods
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 100, "Model predict output length is incorrect."
    assert all(0 <= pred <= 1 for pred in preds), "Model prediction not in range [0, 1]"


def test_notfittederror_prediction():
    model = Model()
    X = np.random.normal(size=(100, 10))

    # Test that NotFittedError is raised when trying to predict before fitting the model
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_notfittederror_predict_proba():
    model = Model()
    X = np.random.normal(size=(100, 10))

    # Test that NotFittedError is raised when trying to predict_proba before fitting the model
    with pytest.raises(NotFittedError):
        model.predict_proba(X)


def test_predict_proba():
    model = Model()
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    # Testing fit and predict_proba methods
    model.fit(X, y)
    prob_preds = model.predict_proba(X)

    assert len(prob_preds) == 100, "Model predict_proba output length is incorrect."
    assert all(
        0 <= prob_pred[1] <= 1 for prob_pred in prob_preds
    ), "Model predict_proba prediction not in range [0, 1]"
