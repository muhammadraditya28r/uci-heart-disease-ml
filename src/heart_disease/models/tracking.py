from typing import Any

import mlflow
import mlflow.sklearn

from heart_disease.config import (
    ExperimentConfig,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)


def start_experiment(
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> None:
    """Set the MLflow experiment."""
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)
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
            "multiple_scoring": ",".join(config.multiple_scoring),
            "drop_threshold": config.drop_threshold,
            "use_scaler": config.use_scaler,
            "numeric_missing_indicator": config.numeric_missing_indicator,
            "categorical_missing_indicator": config.categorical_missing_indicator,
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


def _flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool))
    }


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
        mlflow.log_params(_flatten_params(model_params))
        log_metrics(metrics)


def log_grid_search_run(
    model_name: str,
    config: ExperimentConfig,
    param_grid: dict[str, Any],
    best_params: dict[str, Any],
    best_score: float,
) -> None:
    """Log a GridSearchCV experiment to MLflow."""
    start_experiment()

    with start_run(run_name=f"{model_name}-grid-search"):
        log_experiment_config(config)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("search_type", "GridSearchCV")

        if isinstance(param_grid, dict):
            search_params = {
                f"search_{key}": str(value) for key, value in param_grid.items()
            }
        else:
            search_params = {
                f"search_{i}_{key}": str(value)
                for i, grid in enumerate(param_grid)
                for key, value in grid.items()
            }
        mlflow.log_params(search_params)

        mlflow.log_params(
            {
                f"best_{key}": value
                for key, value in _flatten_params(best_params).items()
            }
        )

        mlflow.log_metric("best_cv_score", best_score)


def log_final_evaluation_model(
    model_name: str,
    config: ExperimentConfig,
    model_params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """Log final held-out test evaluation to MLflow."""
    start_experiment()

    with start_run(run_name=f"{model_name}-final-evaluation"):
        log_experiment_config(config)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("evaluation_stage", "final_stage")
        mlflow.log_params(_flatten_params(model_params))
        log_metrics(metrics)
