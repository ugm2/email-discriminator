import logging
import os

from numpy import ndarray
from rich.logging import RichHandler
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("DataPreprocessor")


class Model:
    """
    Classifier model to fit and predict input data.
    """

    def __init__(self):
        self.model = XGBClassifier(eval_metric="logloss")
        self.fitted = False

    def fit(self, X: ndarray, y: ndarray):
        try:
            self.model.fit(X, y)
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
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Error predicting: {e}")
            raise e
