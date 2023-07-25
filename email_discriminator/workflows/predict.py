import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task

from email_discriminator.core.model import DataProcessor

# You might need to set your MLFlow tracking URI here if it's not already set
mlflow.set_tracking_uri("http://localhost:5001/")


@task
def load_data(file_path):
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


@task
def load_pipeline(model_name):
    logger = get_run_logger()
    logger.info(f"Loading model {model_name}")
    model_uri = (
        f"models:/{model_name}/Production"  # loading the model in 'Production' stage
    )
    pipeline = mlflow.pyfunc.load_model(model_uri)
    return pipeline


@task
def predict(pipeline, data):
    logger = get_run_logger()
    logger.info("Making predictions")
    return pipeline.predict(data)


@flow
def predict_flow():
    logger = get_run_logger()
    logger.info("Starting prediction flow")
    df = load_data("data/tldr_articles.csv")
    pipeline = load_pipeline("email_discriminator")
    predictions = predict(pipeline, df)
    logger.info(predictions)


if __name__ == "__main__":
    predict_flow()
