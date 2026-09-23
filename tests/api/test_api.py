from fastapi.testclient import TestClient

from heart_disease.api import app



client = TestClient(app)


VALID_PAYLOAD = {
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalch": 150,
    "oldpeak": 2.3,
    "sex": "Male",
    "cp": "typical angina",
    "fbs": None,
    "restecg": "lv hypertrophy",
    "exang": 0,
    "slope": "downsloping",
    "ca": 0,
    "thal": "fixed defect",
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint() -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {0, 1}
    assert 0.0 <= data["prediction"] <= 1.0


def test_prediction_endpoint_rejects_invalid_requests() -> None:
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload.pop("age")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422