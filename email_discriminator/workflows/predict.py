import os
from typing import Dict, Union

import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from pandas import DataFrame
from prefect import flow, get_run_logger, task

# Configuration Management
# Fetching configurations from environment variables
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000/")
DATA_PATH = os.getenv("DATA_PATH", "data/tldr_articles.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "email_discriminator")

mlflow.set_tracking_uri(MLFLOW_URI)


@task
def load_data(file_path: str) -> DataFrame:
    """
    Load data from a CSV file.
    """
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


@task
def load_pipeline(model_name: str) -> PythonModel:
    """
    Load a model pipeline from MLFlow.
    """
    logger = get_run_logger()
    logger.info(f"Loading model {model_name}")
    model_uri = (
        f"models:/{model_name}/Production"  # loading the model in 'Production' stage
    )
    pipeline = mlflow.pyfunc.load_model(model_uri)
    return pipeline


@task
def predict(pipeline: PythonModel, data: DataFrame) -> pd.Series:
    """
    Make predictions using a model pipeline and data.
    """
    logger = get_run_logger()
    logger.info("Making predictions")
    return pipeline.predict(data)


@flow
def predict_flow() -> None:
    """
    The main flow for loading data, loading a model, and making predictions.
    """
    logger = get_run_logger()
    logger.info("Starting prediction flow")
    df = load_data(DATA_PATH)
    pipeline = load_pipeline(MODEL_NAME)
    predictions = predict(pipeline, df)
    logger.info(predictions)


if __name__ == "__main__":
    predict_flow()
