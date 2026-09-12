from typing import Any

import mlflow
import mlflow.sklearn

from heart_disease.config import ExperimentConfig


def start_experiment(
    experiment_name: str = "uci-heart-disease",
) -> None:
    """Set the MLflow experiment."""
    mlflow.set_experiment(experiment_name=experiment_name)


def start_run(run_name: str | None = None):
    """Start an MLflow run."""
    return mlflow.start_run(run_name=run_name)


def log_experiment_config(config: ExperimentConfig) -> None:
    """Log experiment configuration to MLflow."""
    mlflow.log_params(
        {
            "random_state": config.random_state,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "single_scoring": config.single_scoring,
            "multiple_scoring": config.multiple_scoring,
            "drop_threshold": config.drop_threshold,
            "use_scaler": config.use_scaler,
            "numeric_missing_indicator": config.numeric_missing_indicator,
            "categorical_missing_indicator": config.categorical_missing_indicator,
            "logistic_regression_params": config.logistic_regression_params,
        }
    )


def log_metrics(metrics: dict[str, float]) -> None:
    """Log a collection of metrics."""
    mlflow.log_metrics(metrics)


def log_model(model: Any, artifact_path: str = "model") -> None:
    """Log a scikit-learn model to MLflow."""
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def log_best_params(params: dict[str, Any]) -> None:
    """Log model hyperparameters."""
    mlflow.log_params(params)


def log_model_comparison_run(
    model_name: str,
    config: ExperimentConfig,
    model_params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log a model comparison experiment to MLflow."""
    start_experiment()

    with start_run(run_name=model_name):
        log_experiment_config(config)
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model_params)
        log_metrics(metrics)
