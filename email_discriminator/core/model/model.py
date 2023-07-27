import logging
import os

import numpy as np
from numpy import ndarray
from rich.logging import RichHandler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("DataPreprocessor")


class Model(BaseEstimator, ClassifierMixin):
    """
    Classifier model to fit and predict input data.
    """

    def __init__(
        self, model=None, threshold=0.5, thresholds=np.arange(0, 1, 0.01), min_f1=0.75
    ):
        self.model = (
            model if model is not None else XGBClassifier(eval_metric="logloss")
        )
        self.threshold = threshold
        self.thresholds = thresholds
        self.min_f1 = min_f1
        self.fitted = False

    def fit(self, X: ndarray, y: ndarray):
        try:
            self.model.fit(X, y)
            # Find the best threshold for recall
            if len(self.thresholds) > 1:
                y_proba_pos = self.model.predict_proba(X)[:, 1]
                best_recall = 0
                for threshold in self.thresholds:
                    y_pred = (y_proba_pos >= threshold).astype(int)
                    f1 = f1_score(y, y_pred, average=None)
                    recall = recall_score(y, y_pred, pos_label=1)
                    if recall > best_recall and min(f1) > self.min_f1:
                        self.threshold = threshold
                        best_recall = recall
            self.fitted = True
        except Exception as e:
            logging.error(f"Error fitting the model: {e}")
            raise e

    def predict_proba(self, X: ndarray) -> ndarray:
        if not self.fitted:
            raise NotFittedError(
                "Model instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logging.error(f"Error predicting probabilities: {e}")
            raise e

    def predict(self, X: ndarray) -> ndarray:
        if not self.fitted:
            raise NotFittedError(
                "Model instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        try:
            y_proba = self.predict_proba(X)
            y_proba_pos = y_proba[:, 1]
            return (y_proba_pos >= self.threshold).astype(int)
        except Exception as e:
            logging.error(f"Error predicting: {e}")
            raise e

    def get_params(self, deep=True):
        return {"model": self.model, "threshold": self.threshold}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def classes_(self):
        return self.model.classes_
