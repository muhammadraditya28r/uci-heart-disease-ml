from dataclasses import dataclass
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from pathlib import Path
import joblib
from collections.abc import Sequence
from typing import Any
from numpy.typing import NDArray
from heart_disease.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    random_state: int = 42
    cv: int | StratifiedKFold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    scoring: str = "f1"


def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Basic training function

    Return:
        Scikit-learn Pipeline
    """
    model.fit(X_train, y_train)
    logger.info(
        "Training: %s with shape of dataframe: %d", type(model).__name__, X_train.shape
    )

    return model


def cross_validate_model(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, config: TrainingConfig
) -> NDArray:
    """
    Performing cross validation

    Return:
        Model evaluation metrics in NDArray
    """
    logger.info(
        "Starting cross validation for %s model with cv: %d and scoring: %s",
        type(model).__name__,
        config.cv,
        config.scoring,
    )
    scores = cross_val_score(
        estimator=model, X=X, y=y, cv=config.cv, scoring=config.scoring
    )
    logger.info(
        "Finishing cross validation with scores: CV F1: %.3f ± %.3f",
        scores.mean(),
        scores.std(),
    )
    return scores


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
        scoring=config.scoring,
        n_jobs=-1,
    )

    search.fit(X, y)

    logger.info(
        "Best parameters: %s | Best CV scores: %.3f",
        search.best_params_,
        search.best_score_,
    )
    return search


def save_model(model: Pipeline, path: Path) -> None:
    """
    A Function to save a model to corresponding path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving %s model to %s", type(model).__name__, path)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """
    A Function to load a model from corresponding path
    """
    model = joblib.load(path)
    logger.info("Loading %s model from %s", type(model).__name__, path)
    return model
