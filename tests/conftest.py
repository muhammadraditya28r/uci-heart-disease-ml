import pandas as pd
import pytest

from heart_disease.config import FEATURES, TARGET_COLUMN, RAW_DATA_DIR
from heart_disease.data.ingestion import load_file
from heart_disease.features.cleaning import clean_data


@pytest.fixture
def sample_data() -> pd.DataFrame:
    df = load_file(path=RAW_DATA_DIR / "heart_disease_uci.csv")
    df = clean_data(df)

    normal = df[df[TARGET_COLUMN] == 0].sample(5, random_state=42)
    positive = df[df[TARGET_COLUMN] > 0].sample(5, random_state=42)
    missing = df[df[FEATURES].isna().any(axis=1)].sample(5, random_state=42)

    return pd.concat([normal, positive, missing]).drop_duplicates()


@pytest.fixture
def sample_features(sample_data: pd.DataFrame) -> pd.DataFrame:
    return sample_data[FEATURES]


@pytest.fixture
def sample_target(sample_data: pd.DataFrame) -> pd.Series:
    return sample_data[TARGET_COLUMN]