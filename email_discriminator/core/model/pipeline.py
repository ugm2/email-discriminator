import logging
import os

from joblib import dump, load
from numpy import ndarray
from pandas import DataFrame
from rich.logging import RichHandler
from sklearn.exceptions import NotFittedError

from email_discriminator.core.model.data_processor import DataProcessor
from email_discriminator.core.model.model import Model

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("DataPreprocessor")


class Pipeline:
    """
    Pipeline to process data and perform predictions.
    """

    def __init__(self, data_processor: DataProcessor, model: Model):
        self.data_processor = data_processor
        self.model = model
        self.fitted = False

    def fit(self, X: DataFrame, y: ndarray):
        try:
            X = self.data_processor.fit_transform(X)
            self.model.fit(X, y)
            self.fitted = True
        except Exception as e:
            logging.error(f"Error fitting the pipeline: {e}")
            raise e

    def predict(self, X: DataFrame) -> ndarray:
        if not self.fitted:
            raise NotFittedError(
                "Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        try:
            X = self.data_processor.transform(X)
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Error predicting with the pipeline: {e}")
            raise e

    def predict_proba(self, X: DataFrame) -> ndarray:
        if not self.fitted:
            raise NotFittedError(
                "Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        try:
            X = self.data_processor.transform(X)
            return self.model.predict_proba(X)
        except Exception as e:
            logging.error(f"Error predicting probabilities with the pipeline: {e}")
            raise e

    def save(self, filename: str):
        """
        Save the trained model as a pickle file.
        """
        if not self.fitted:
            raise NotFittedError(
                "Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )
        dump(self, filename)

    @staticmethod
    def load(filename: str):
        """
        Load a pickled model.
        """
        return load(filename)
