from pathlib import Path

import pandas as pd
import pytest

from heart_disease.data.ingestion import load_file


def test_load_csv(tmp_path: Path) -> None:
    """Test that csv file is loaded correctly."""
    
    data = pd.DataFrame(
        {
            "age": [50, 60],
            "target": [0, 1]
        }
    )

    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)

    result = load_file(file_path)

    pd.testing.assert_frame_equal(result, data)


def test_load_csv_raises_error_for_missing_file() -> None:
    """Test that missing file raises FileNotFoundError"""

    with pytest.raises(FileNotFoundError):
        load_file("does_not_exist.csv")