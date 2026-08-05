from __future__ import annotations

from heart_disease.utils.logging import get_logger

import numpy as np
import pandas as pd


logger = get_logger(__name__)


def replace_zero_features_to_nan(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Replace features that impossible to has value zero or less with NaN
    """

    cleaned = df.copy()

    for column in columns:
        try:
            invalid = cleaned[column] <= 0
            cleaned.loc[invalid, "chol"] = np.nan
        except KeyError:
            logger.error("Mismatched columns name")
        else:
            logger.info("Replacing %d invalid %s values.", invalid.sum(), column)
        finally:
            return cleaned


def target_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert target from multiclass to binary
    """

    cleaned = df.copy()

    try:
        cleaned["target"] = (cleaned["target"] > 0).astype(int)
    except KeyError:
        logger.error("There is no column named target")
    else:
        logger.info("Converting target to binary")
    finally:
        return cleaned


def clean_data(
    df: pd.DataFrame,
    *,
    column_mapping: dict = {"num": "target"},
    replace_zero_with_nan: list[str] = ["chol", "trestbps", "oldpeak"],
    convert_target_to_binary: bool = True,
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

    if column_mapping:
        cleaned = cleaned.rename(columns=column_mapping)

    if replace_zero_with_nan:
        cleaned = replace_zero_features_to_nan(cleaned, replace_zero_with_nan)

    if convert_target_to_binary:
        cleaned = target_to_binary(cleaned)
