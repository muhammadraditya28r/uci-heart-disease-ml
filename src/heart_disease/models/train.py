from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
import pandas as pd
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from pathlib import Path
import joblib
from collections.abc import Sequence
from typing import Any
from numpy.typing import NDArray
from heart_disease.utils.logging import get_logger
from heart_disease.config import RANDOM_STATE, TARGET_COLUMN, FEATURES
from collections.abc import Callable

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    random_state: int = RANDOM_STATE
    cv: int | StratifiedKFold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    single_scoring: str = "f1"
    multiple_scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]


def split_data(
    df: pd.DataFrame,
    target_column: str,
    features_column: Sequence[Any],
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    A function for split data into train and test

    Returns:
        X and y for train and test in pandas dataframe
    """
    y = df[target_column]
    X = df[features_column]

    stratify_data = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_data
    )
    logger.info(
        "Created with shape of X_train: %s, X_test: %s, y_train: %s, y_test: %s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )

    return X_train, X_test, y_train, y_test


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
        "Training: %s with shape of dataframe: %s", type(pipeline.named_steps["classifier"]).__name__, X_train.shape
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
        config.cv,
        config.single_scoring,
    )
    scores = cross_val_score(
        estimator=pipeline, X=X, y=y, cv=config.cv, scoring=config.single_scoring
    )
    logger.info(
        "Finishing cross validation with scores: CV F1: %.3f ± %.3f",
        scores.mean(),
        scores.std(),
    )

    df = {f'fold {num + 1}': scores[num] for num in range(len(scores))}
    df['mean'] = scores.mean()
    df['std'] = scores.std()

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
        cv=config.cv,
        scoring=config.single_scoring,
        n_jobs=-1,
    )

    search.fit(X, y)

    logger.info(
        "Best parameters: %s | Best CV scores: %.3f",
        search.best_params_,
        search.best_score_,
    )
    return search


def model_comparison_cv(
    models: list[ClassifierMixin], pipeline_factory: Callable[[ClassifierMixin], Pipeline], X: pd.DataFrame, y: pd.Series, config: TrainingConfig,
) -> pd.DataFrame:
    result = []

    for model in models:
        model_pipeline = pipeline_factory(model)
        scores = cross_validate(model_pipeline, X, y, cv=config.cv, scoring=config.multiple_scoring)

        result.append(
            {
                "Model": type(model_pipeline.named_steps["classifier"]).__name__,
                "fit_time": round(scores["fit_time"].mean(), 4),
                "Accuracy": round(scores["test_accuracy"].mean(), 4),
                "Precision": round(scores["test_precision"].mean(), 4),
                "Recall": round(scores["test_recall"].mean(), 4),
                "F1": round(scores["test_f1"].mean(), 4),
                "ROC AUC": round(scores["test_roc_auc"].mean(), 4),
            }
        )

    return (pd.DataFrame(result).sort_values("F1", ascending=False).reset_index(drop=True)
)

def save_model(pipeline: Pipeline, path: Path) -> None:
    """
    A Function to save a model to corresponding path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s model to %s", type(pipeline.named_steps["classifier"]).__name__, path)
    joblib.dump(pipeline, path)


def load_model(path: Path) -> Pipeline:
    """
    A Function to load a model from corresponding path
    """
    pipeline = joblib.load(path)
    logger.info("Loading %s model from %s", type(pipeline.named_steps["classifier"]).__name__, path)
    return pipeline
