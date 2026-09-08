from __future__ import annotations

import numpy as np
import pandas as pd

from heart_disease.utils.logging import get_logger
from heart_disease.config import FULL_COLUMNS


logger = get_logger(__name__)


def calculate_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage of missing data in each columns"""
    return pd.DataFrame(df.isna().mean().sort_values(ascending=False) * 100).rename(
        columns={0: "Percentage of missing data"}
    )


def replace_invalid_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace non-positive values in the specified columns with NaN.
    """

    cleaned = df.copy()

    for column in columns:
        if column not in cleaned.columns:
            logger.warning("Column %s not found. Skipping.", column)
            continue

        invalid = cleaned[column] <= 0
        cleaned.loc[invalid, column] = np.nan

        if invalid.any():
            logger.info(
                "Replacing %d non-positive %s values to nan.", invalid.sum(), column
            )

    return cleaned


def target_to_binary(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """
    Convert target from multiclass to binary
    """

    cleaned = df.copy()

    if target is None:
        target = "target"

    if target not in cleaned.columns:
        logger.error("There is no column named target")
        return cleaned

    cleaned[target] = (cleaned[target] > 0).astype(int)
    logger.info("Converting target to binary")

    return cleaned


def clean_data(
    df: pd.DataFrame,
    *,
    rename_columns: dict[str, str] | None = None,
    invalid_value_columns: list[str] | None = None,
    convert_target_to_binary: bool = True,
    drop_thresh: int | None = None,
    drop_unused_cols: bool = True,
) -> pd.DataFrame:
    """
    Execute the complete cleaning pipeline.

    Parameters
    ----------
    df
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    cleaned = df.copy()

    if rename_columns is None:
        rename_columns = {"num": "target"}
    if invalid_value_columns is None:
        invalid_value_columns = ["chol", "trestbps", "oldpeak"]

    cleaned = cleaned.rename(columns=rename_columns)
    cleaned = replace_invalid_values(cleaned, invalid_value_columns)
    if drop_unused_cols:
        cleaned = cleaned[FULL_COLUMNS]
    if convert_target_to_binary:
        cleaned = target_to_binary(cleaned)
    if drop_thresh is not None:
        rows = len(cleaned.index)
        cleaned = cleaned.dropna(thresh=drop_thresh)
        logger.info(
            "Threshold: %d, deleting %d rows", drop_thresh, rows - len(cleaned.index)
        )

    return cleaned
