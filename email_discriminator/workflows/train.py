import mlflow
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imblearnPipeline
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from email_discriminator.core.model import DataProcessor, Model

mlflow.set_tracking_uri("http://localhost:5001/")
mlflow.get_experiment_by_name("email_discriminator")


@task
def load_data(file_path):
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


@task
def split_data(df):
    logger = get_run_logger()
    logger.info("Splitting data into train and test sets")
    return train_test_split(
        df[["article", "section"]], df["is_relevant"], test_size=0.2, random_state=42
    )


@task
def load_pipeline():
    logger = get_run_logger()
    logger.info("Loading pipeline")
    return imblearnPipeline(
        [
            ("features", DataProcessor()),
            ("sampling", RandomOverSampler()),
            ("model", Model()),
        ]
    )


@task
def create_grid_search(pipeline):
    logger = get_run_logger()
    logger.info("Creating GridSearchCV object")
    scorer = make_scorer(recall_score)
    model_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }
    return GridSearchCV(pipeline, model_params, cv=3, scoring=scorer)


@task
def fit(grid_search, X_train, y_train):
    logger = get_run_logger()
    logger.info("Fitting GridSearchCV object")
    grid_search.fit(X_train, y_train)


@task
def evaluation(grid_search, X_test, y_test):
    logger = get_run_logger()
    logger.info("Evaluating the best model")
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


@task
def log_metrics_and_model(report, grid_search, model_name):
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

    # Log model version in prefect as an artifact
    model_version_details = {
        "name": model_version.name,
        "version": model_version.version,
        "creation_timestamp": model_version.creation_timestamp,
        "last_updated_timestamp": model_version.last_updated_timestamp,
        "current_stage": model_version.current_stage,
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


@flow
def train():
    logger = get_run_logger()
    logger.info("Starting training flow")
    df = load_data("data/tldr_articles.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = load_pipeline()
    grid_search = create_grid_search(pipeline)
    fit(grid_search, X_train, y_train)
    report = evaluation(grid_search, X_test, y_test)
    log_metrics_and_model(report, grid_search, "email_discriminator")


if __name__ == "__main__":
    train()
