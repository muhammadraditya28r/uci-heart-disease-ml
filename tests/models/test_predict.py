from heart_disease.config import PRODUCTION_MODEL_DIR
from heart_disease.models.train import load_model
from heart_disease.models.predict import predict


def test_production_model_prediction() -> None:
    model = load_model(PRODUCTION_MODEL_DIR)

    features = {
        "age": 63,
        "trestbps": 145,
        "chol": 233,
        "thalch": 150,
        "oldpeak": 2.3,
        "sex": 1,
        "cp": "typical angina",
        "fbs": None,
        "restecg": "lv hypertrophy",
        "exang": 0,
        "slope": "downsloping",
        "ca": 0,
        "thal": "fixed defect",
    }

    prediction, probability = predict(model, features)

    assert prediction in {0, 1}
    assert 0.0 <= probability <= 1.0
