from collections.abc import Sequence
from typing import Any

from sklearn.model_selection import train_test_split
import pandas as pd

from heart_disease.config import RANDOM_STATE
from heart_disease.utils.logging import get_logger


logger = get_logger(__name__)


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
