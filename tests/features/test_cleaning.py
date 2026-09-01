import numpy as np
import pandas as pd

from heart_disease.features.cleaning import (
    replace_invalid_values,
    target_to_binary,
)


def test_replace_invalid_values() -> None:
    df = pd.DataFrame(
        {
            "chol": [200, 0, 250, -10],
        }
    )

    result = replace_invalid_values(df, ["chol"])

    expected = pd.DataFrame(
        {
            "chol": [200, np.nan, 250, np.nan],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_replace_invalid_values_does_not_modify_original() -> None:
    df = pd.DataFrame({"chol": [200, 0, 250]})
    original = df.copy()

    replace_invalid_values(df, ["chol"])

    pd.testing.assert_frame_equal(df, original)


def test_replace_invalid_values_skips_missing_column() -> None:
    df = pd.DataFrame({"chol": [200, 0, 250]})

    result = replace_invalid_values(
        df,
        ["chol", "does_not_exist"],
    )

    expected = pd.DataFrame({"chol": [200, np.nan, 250]})

    pd.testing.assert_frame_equal(result, expected)


def test_target_to_binary() -> None:
    df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    result = target_to_binary(df)

    expected = pd.DataFrame({"target": [0, 1, 1, 1, 1]})

    pd.testing.assert_frame_equal(result, expected)


def test_target_to_binary_does_not_modify_original() -> None:
    df = pd.DataFrame({"target": [0, 1, 2, 3]})
    original = df.copy()

    target_to_binary(df)

    pd.testing.assert_frame_equal(df, original)