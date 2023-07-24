import logging
import os

from numpy import ndarray
from rich.logging import RichHandler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("LabelEncoder")


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom LabelEncoder that extends BaseEstimator and TransformerMixin.
    """

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X: ndarray, y: ndarray = None) -> "CustomLabelEncoder":
        try:
            self.encoder.fit(X)
            return self
        except Exception as e:
            logging.error(f"Error fitting data: {e}")
            raise e

    def transform(self, X: ndarray) -> ndarray:
        try:
            return self.encoder.transform(X).reshape(-1, 1)
        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            raise e
