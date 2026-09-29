import pytest

from heart_disease.config import PRODUCTION_MODEL_DIR
from heart_disease.models.train import load_model
from heart_disease.models.predict import predict


@pytest.fixture
def test_production_model_prediction(sample_features) -> None:
    model = load_model(PRODUCTION_MODEL_DIR)

    prediction, probability = predict(model, sample_features)

    assert prediction in {0, 1}
    assert 0.0 <= probability <= 1.0
