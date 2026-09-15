from collections.abc import Sequence
from typing import Any
from collections.abc import Callable

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from pathlib import Path
import joblib

from heart_disease.utils.logging import get_logger
from heart_disease.config import ExperimentConfig
from heart_disease.models.tracking import log_model_comparison_run, log_grid_search_run


logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    experiment: ExperimentConfig

    @property
    def cv_skfold(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.experiment.cv_folds,
            shuffle=True,
            random_state=self.experiment.random_state,
        )

    @property
    def cv(self) -> int:
        return self.experiment.cv_folds

    @property
    def single_scoring(self) -> str:
        return self.experiment.single_scoring

    @property
    def multiple_scoring(self) -> list[str]:
        return self.experiment.multiple_scoring


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Basic training function

    Return:
        Scikit-learn Pipeline
    """
    pipeline.fit(X_train, y_train)
    logger.info(
        "Training: %s with shape of dataframe: %s",
        type(pipeline.named_steps["classifier"]).__name__,
        X_train.shape,
    )

    return pipeline


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, config: TrainingConfig
) -> pd.DataFrame:
    """
    Performing cross validation

    Return:
        Model evaluation metrics in NDArray
    """
    logger.info(
        "Starting cross validation for %s model with cv: %s and scoring: %s",
        type(pipeline.named_steps["classifier"]).__name__,
        config.cv_skfold,
        config.single_scoring,
    )
    scores = cross_val_score(
        estimator=pipeline, X=X, y=y, cv=config.cv_skfold, scoring=config.single_scoring
    )
    logger.info(
        "Finishing cross validation with scores: CV F1: %.3f ± %.3f",
        scores.mean(),
        scores.std(),
    )

    df = {f"fold {num + 1}": scores[num] for num in range(len(scores))}
    df["mean"] = scores.mean()
    df["std"] = scores.std()

    return pd.DataFrame([df])


def grid_search(
    model: Pipeline,
    param_grid: dict[str, Sequence[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> GridSearchCV:
    """
    Performing grid search for finding model's best parameters

    Return:
        GridSearchCV
    """
    logger.info("Starting grid search")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=config.cv_skfold,
        scoring=config.single_scoring,
        n_jobs=-1,
    )

    search.fit(X, y)

    log_grid_search_run(
        model_name=model.__class__.__name__,
        config=config.experiment,
        param_grid=param_grid,
        best_params=search.best_params_,
        best_score=search.best_score_,
    )

    logger.info(
        "Best parameters: %s | Best CV scores: %.3f",
        search.best_params_,
        search.best_score_,
    )
    return search


def model_comparison_cv(
    models: list[ClassifierMixin],
    pipeline_factory: Callable[[ClassifierMixin], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Compare multiple classifiers using cross-validation."""

    results: list[dict[str, Any]] = []

    for model in models:
        model_name = type(model).__name__

        logger.info(
            "Starting cross-validation for %s",
            model_name,
        )

        model_pipeline = pipeline_factory(model)

        scores = cross_validate(
            estimator=model_pipeline,
            X=X,
            y=y,
            cv=config.cv_skfold,
            scoring=config.multiple_scoring,
            n_jobs=-1,
        )

        metrics = {
            "fit_time": scores["fit_time"].mean(),
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
        }

        log_model_comparison_run(
            model_name=model_name,
            config=config.experiment,
            model_params=model.get_params(),
            metrics=metrics,
        )

        results.append(
            {
                "Model": model_name,
                "fit_time": metrics["fit_time"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC AUC": metrics["roc_auc"],
            }
        )

        logger.info(
            "Finished %s: F1 = %.4f ± %.4f",
            model_name,
            scores["test_f1"].mean(),
            scores["test_f1"].std(),
        )

    return (
        pd.DataFrame(results)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
        .round(4)
    )


def save_model(pipeline: Pipeline, path: Path) -> None:
    """
    A Function to save a model to corresponding path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Saving %s model to %s", type(pipeline.named_steps["classifier"]).__name__, path
    )
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    """
    A Function to load a model from corresponding path
    """
    pipeline = joblib.load(path)
    logger.info(
        "Loading %s model from %s",
        type(pipeline.named_steps["classifier"]).__name__,
        path,
    )
    return pipeline
