import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression

from heart_disease.features.preprocessing import create_training_pipeline
from heart_disease.models.evaluation import (
    evaluate_model,
    get_classification_report,
    get_confusion_matrix,
)


@pytest.fixture
def fitted_model(
    sample_features: pd.DataFrame,
    sample_target: pd.Series,
):
    model = create_training_pipeline(
        model=LogisticRegression()
    )

    model.fit(sample_features, sample_target)

    return model, sample_features, sample_target

def test_evaluate_model(fitted_model) -> None:
    model, X, y = fitted_model

    metrics = evaluate_model(model, X, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_evaluation_metrics_are_valid(fitted_model) -> None:
    model, X, y = fitted_model

    metrics = evaluate_model(model, X, y)

    for value in metrics.values():
        assert 0 <= value <= 1


def test_get_confusion_matrix(fitted_model) -> None:
    model, X, y = fitted_model

    result = get_confusion_matrix(model, X, y)

    assert result.shape == (2, 2)
    assert list(result.index) == ["Actual 0", "Actual 1"]
    assert list(result.columns) == ["Predicted 0", "Predicted 1"]


def test_get_classification_report(fitted_model) -> None:
    model, X, y = fitted_model

    result = get_classification_report(model, X, y)

    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "f1-score" in result.columns