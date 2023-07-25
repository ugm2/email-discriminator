import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as imblearnPipeline
from prefect import flow, task
from rich import print
from sklearn.metrics import classification_report, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from email_discriminator.core.model import DataProcessor, Model


@task
def load_data(file_path):
    return pd.read_csv(file_path)


@task
def split_data(df):
    return train_test_split(
        df[["article", "section"]], df["is_relevant"], test_size=0.2, random_state=42
    )


@task
def load_pipeline():
    return imblearnPipeline(
        [
            ("features", DataProcessor()),
            ("sampling", RandomOverSampler()),
            ("model", Model()),
        ]
    )


@task
def create_grid_search(pipeline):
    # Define scorer based on recall
    scorer = make_scorer(recall_score)
    # Define GridSearchCV
    model_params = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1, 0.2],
    }
    return GridSearchCV(pipeline, model_params, cv=3, scoring=scorer)


@task
def fit(grid_search, X_train, y_train):
    grid_search.fit(X_train, y_train)


@task
def log_metrics(grid_search, X_test, y_test):
    # Make predictions with the best model
    y_pred = grid_search.predict(X_test)
    # Compute metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification report:")
    print(report)
    print("\nBest params:")
    print(grid_search.best_params_)


@flow
def train():
    df = load_data("data/tldr_articles.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = load_pipeline()
    grid_search = create_grid_search(pipeline)
    fit(grid_search, X_train, y_train)
    log_metrics(grid_search, X_test, y_test)


if __name__ == "__main__":
    train()
