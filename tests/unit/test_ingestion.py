from pathlib import Path

import pandas as pd
import pytest

from heart_disease.data.ingestion import load_csv


def test_load_csv(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "age": [50, 60],
            "target": [0, 1]
        }
    )

    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)

    result = load_csv(file_path)

    pd.testing.assert_frame_equal(result, data)


def test_load_csv_raises_error_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_csv("does_not_exist.csv")