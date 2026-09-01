from __future__ import annotations


import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from heart_disease.utils.logging import get_logger


logger = get_logger(__name__)


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a fitted classification model on the test set.

    The model must already be fitted.
    """

    logger.info(
        "Evaluating %s on test data.", type(model.named_steps["classifier"]).__name__
    )

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_pred, y_test),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    logger.info(
        "Test metrics | Accuracy: %.3f | Precision: %.3f | "
        "Recall: %.3f | F1: %.3f | ROC-AUC: %.3f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics.get("roc_auc", float("nan")),
    )

    return metrics


def get_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Return the confusion matrix as a DataFrame
    """

    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)

    return pd.DataFrame(
        matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
    )


def get_classification_report(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Return a classification report as a DataFrame
    """

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    return pd.DataFrame(report).transpose()
