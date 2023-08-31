import logging
import os

from numpy import ndarray
from pandas import DataFrame
from rich.logging import RichHandler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from email_discriminator.core.model.label_encoder import CustomLabelEncoder

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("DataPreprocessor")


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single text column from the data frame to perform additional transformations on.
    """

    def __init__(self, key: str):
        self.key = key

    def fit(self, X: DataFrame, y: ndarray = None) -> "TextSelector":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if self.key not in X.columns:
            raise ValueError(
                "Column `{}` not found! Available columns: {}".format(
                    self.key, ", ".join(X.columns)
                )
            )
        return X[self.key]


class DataProcessor(BaseEstimator, TransformerMixin):
    """
    Data processor to fit and transform input data.
    """

    def __init__(self):
        self.text = Pipeline(
            [
                ("selector", TextSelector(key="article")),
                ("tfidf", TfidfVectorizer(stop_words="english")),
            ]
        )
        self.section = Pipeline(
            [
                ("selector", TextSelector(key="section")),
                ("encoder", CustomLabelEncoder()),
            ]
        )
        self.features = FeatureUnion(
            [("article", self.text), ("section", self.section)]
        )

    def fit(self, X: DataFrame, y: ndarray = None):
        if X.empty:
            raise ValueError("Input DataFrame is empty!")

        try:
            self.features.fit(X, y)
        except Exception as e:
            logging.error(f"Error fitting data: {e}")
            raise e

        return self

    def transform(self, X: DataFrame) -> ndarray:
        try:
            return self.features.transform(X)
        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            raise e

    def fit_transform(self, X: DataFrame, y: ndarray = None):
        self.fit(X, y)
        return self.transform(X)
