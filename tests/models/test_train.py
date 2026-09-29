import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart_disease.features.preprocessing import create_training_pipeline
from heart_disease.models.train import train_model


@pytest.fixture
def test_train_model(sample_features: pd.DataFrame, sample_target: pd.Series) -> None:
    model = create_training_pipeline(model=LogisticRegression())

    result = train_model(model, sample_features, sample_target)

    assert result is not None
