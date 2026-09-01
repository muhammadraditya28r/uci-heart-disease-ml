import pandas as pd
import pytest


@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [40, 45, 60, 65, 70, 75],
            "trestbps": [120, 125, 130, 135, 140, 145],
            "chol": [180, 200, 220, 240, 260, 280],
            "thalch": [170, 165, 155, 150, 140, 130],
            "oldpeak": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "sex": [0, 1, 0, 1, 1, 0],
            "cp": [1, 2, 1, 3, 2, 3],
            "fbs": [0, 0, 1, 0, 1, 0],
            "restecg": [0, 1, 0, 1, 0, 1],
            "exang": [0, 0, 1, 1, 1, 1],
            "slope": [1, 2, 1, 2, 2, 3],
            "ca": [0, 0, 1, 1, 2, 2],
            "thal": [3, 3, 6, 6, 7, 7],
        }
    )


@pytest.fixture
def sample_target() -> pd.Series:
    return pd.Series([0, 0, 0, 1, 1, 1])
