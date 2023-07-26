import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, recall_score

from email_discriminator.core.model import Model


def test_fit_predict():
    model = Model()
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    # Testing fit and predict methods
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == 100, "Model predict output length is incorrect."
    assert all(0 <= pred <= 1 for pred in preds), "Model prediction not in range [0, 1]"

    # Testing the thresholding in predict
    y_proba = model.predict_proba(X)[:, 1]
    y_pred_threshold = (y_proba >= model.threshold).astype(int)
    assert np.array_equal(
        preds, y_pred_threshold
    ), "Model prediction not consistent with threshold."


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


def test_threshold_finding():
    model = Model(thresholds=np.arange(0, 1, 0.01), min_f1=0.7)
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    # Testing fit and predict methods
    model.fit(X, y)
    y_pred = model.predict(X)

    assert (
        recall_score(y, y_pred, pos_label=1) >= 0.7
    ), "Recall score not meeting threshold."
    assert (
        min(f1_score(y, y_pred, average=None)) >= 0.7
    ), "F1 score not meeting threshold."


def test_get_params():
    model = Model()
    params = model.get_params()

    assert isinstance(params, dict), "get_params should return a dictionary."
    assert "model" in params, "get_params should include 'model'."
    assert "threshold" in params, "get_params should include 'threshold'."


def test_set_params():
    model = Model()
    model.set_params(model=None, threshold=0.6)

    assert model.model is None, "set_params should correctly set the 'model' attribute."
    assert (
        model.threshold == 0.6
    ), "set_params should correctly set the 'threshold' attribute."


def test_classes_():
    model = Model()
    X = np.random.normal(size=(100, 10))
    y = np.random.choice([0, 1], size=100)

    model.fit(X, y)

    assert hasattr(
        model, "classes_"
    ), "model should have 'classes_' attribute after fitting."
    assert np.array_equal(
        model.classes_, np.array([0, 1])
    ), "'classes_' should be correctly set after fitting."
