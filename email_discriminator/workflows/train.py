import io
import os
from typing import Dict, Optional, Tuple

import mlflow
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imblearnPipeline
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect_alert import alert_on_failure
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from email_discriminator.core.data_versioning import GCSVersionedDataHandler
from email_discriminator.core.model import DataProcessor, Model

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://35.206.147.175:5000")
DATA_PATH = os.getenv("DATA_PATH", "data/tldr_articles.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "email_discriminator")
BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MODEL_NAME)


@task
def load_training_data(gcs_handler: GCSVersionedDataHandler) -> pd.DataFrame:
    """
    Loads the original data and all the training data from GCS.

    Args:
        gcs_handler: GCSVersionedDataHandler instance.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    logger = get_run_logger()
    logger.info("Loading original and training data from GCS")

    # Load original data
    csv_string = gcs_handler.download_original_data()
    original_data = pd.read_csv(io.StringIO(csv_string))
    logger.info("Original data shape: {}".format(original_data.shape))
    logger.debug(original_data.head())

    # Load all training data
    training_data_files = gcs_handler.download_all_training_data()
    logger.info("Number of training data files: {}".format(len(training_data_files)))
    training_data_dfs = []
    for csv in training_data_files.values():
        df = pd.read_csv(io.StringIO(csv))
        df.drop(
            columns=["predicted_is_relevant", "Unnamed: 0"],
            inplace=True,
            errors="ignore",
        )
        logger.info(f"Loaded training data with shape {df.shape}")
        logger.debug(df.head())
        training_data_dfs.append(df)
    logger.info(
        "Number of training data files loaded: {}".format(len(training_data_dfs))
    )

    # Concatenate all dataframes together
    df = pd.concat([original_data] + training_data_dfs, ignore_index=True)

    logger.info(f"Loaded data with shape {df.shape}")
    return df


@task
def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Args:
        df: A pandas DataFrame containing the data to be split.

    Returns:
        A tuple containing the training and testing data and labels.
    """
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    return train_test_split(
        df[["article", "section"]], df["is_relevant"], test_size=0.2, random_state=42
    )


@task
def create_pipeline() -> imblearnPipeline:
    logger = get_run_logger()
    logger.info("Creating pipeline")
    return imblearnPipeline(
        [
            ("features", DataProcessor()),
            ("sampling", RandomOverSampler()),
            ("model", Model()),
        ]
    )


@task
def create_grid_search(pipeline: imblearnPipeline) -> GridSearchCV:
    logger = get_run_logger()
    logger.info("Creating GridSearchCV object")
    scorer = make_scorer(recall_score)
    model_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }
    return GridSearchCV(pipeline, model_params, cv=3, scoring=scorer)


@task
def fit(grid_search: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    logger = get_run_logger()
    logger.info("Fitting GridSearchCV object")
    grid_search.fit(X_train, y_train)


@task
def evaluation(
    grid_search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict:
    logger = get_run_logger()
    logger.info("Evaluating the best model")
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


@task
def log_metrics_and_model(
    report: Dict, grid_search: GridSearchCV, model_name: str, model_stage: Optional[str]
) -> None:
    """
    Logs metrics and model to MLFlow, registers the model, and logs model version as a Prefect artifact.
    """
    logger = get_run_logger()
    logger.info("Logging metrics and model")
    logger.info(report)
    mlflow.log_metric("accuracy", report["accuracy"])
    mlflow.log_metric("precision", report["macro avg"]["precision"])
    mlflow.log_metric("recall", report["macro avg"]["recall"])
    mlflow.log_metric("f1-score", report["macro avg"]["f1-score"])

    class_labels = grid_search.best_estimator_.named_steps["model"].classes_
    table_data = []
    for i, class_label in enumerate(class_labels):
        mlflow.log_metric(f"{class_label}_precision", report[str(i)]["precision"])
        mlflow.log_metric(f"{class_label}_recall", report[str(i)]["recall"])
        mlflow.log_metric(f"{class_label}_f1-score", report[str(i)]["f1-score"])
        row_data = {
            "Metric": f"Class {class_label}",
            "Precision": round(report[str(i)]["precision"], 2),
            "Recall": round(report[str(i)]["recall"], 2),
            "F1-Score": round(report[str(i)]["f1-score"], 2),
            "Support": int(report[str(i)]["support"]),
        }
        table_data.append(row_data)

    for key in ["macro avg", "weighted avg"]:
        row_data = {
            "Metric": key,
            "Precision": round(report[key]["precision"], 2),
            "Recall": round(report[key]["recall"], 2),
            "F1-Score": round(report[key]["f1-score"], 2),
            "Support": int(report[key]["support"]),
        }
        table_data.append(row_data)

    accuracy_row = {
        "Metric": "accuracy",
        "Precision": "",
        "Recall": "",
        "F1-Score": "",
        "Support": report["accuracy"],
    }
    table_data.append(accuracy_row)

    create_table_artifact(
        key="classification-report",
        table=table_data,
        description="Classification Report",
    )

    logger.info(grid_search.best_params_)
    mlflow.log_params(grid_search.best_params_)
    mlflow.set_tag("model_name", model_name)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    # Register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    client = mlflow.tracking.MlflowClient()
    if model_stage is not None:
        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage=model_stage
        )

    # Log model version in prefect as an artifact
    model_version_details = {
        "name": model_version.name,
        "version": model_version.version,
        "creation_timestamp": model_version.creation_timestamp,
        "last_updated_timestamp": model_version.last_updated_timestamp,
        "current_stage": model_version.status if model_stage is None else model_stage,
        "description": model_version.description,
        "user_id": model_version.user_id,
        "source": model_version.source,
        "run_id": model_version.run_id,
        "status": model_version.status,
        "status_message": model_version.status_message,
    }

    # Convert to a list of dictionaries for the table artifact
    table = [model_version_details]

    create_table_artifact(
        key="model-version-info", table=table, description="Model version information"
    )


@alert_on_failure(to=["unaigaraymaestre@gmail.com"])
@flow(name="train-flow")
def train_flow(model_stage: Optional[str]) -> None:
    """
    The main flow for training the model, includes loading data, splitting it, creating and fitting a pipeline,
    evaluating the pipeline, and logging metrics and model.
    """
    logger = get_run_logger()
    logger.info("Starting training flow")

    # Create a GCSVersionedDataHandler instance
    gcs_handler = GCSVersionedDataHandler(BUCKET_NAME)

    # Load and split the data
    df = load_training_data(gcs_handler)
    X_train, X_test, y_train, y_test = split_data(df)

    # Create pipeline and grid search
    pipeline = create_pipeline()
    grid_search = create_grid_search(pipeline)

    # Train
    fit(grid_search, X_train, y_train)

    # Evaluate
    report = evaluation(grid_search, X_test, y_test)

    # Log metrics and model
    log_metrics_and_model(report, grid_search, MODEL_NAME, model_stage)


if __name__ == "__main__":
    train_flow()
