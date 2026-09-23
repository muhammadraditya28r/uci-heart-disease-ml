from fastapi import FastAPI
from pydantic import BaseModel

from heart_disease.config import PRODUCTION_MODEL_DIR
from heart_disease.models.predict import predict
from heart_disease.models.train import load_model


app = FastAPI(
    title="Heart Disease Prediction API",
    version="0.1.0",
    description="API fro predicting heart disease from clinical feature.",
)


model = load_model(PRODUCTION_MODEL_DIR)


class PredictionRequest(BaseModel):
    age: int
    trestbps: float
    chol: float
    thalch: float
    oldpeak: float
    sex: str
    cp: str
    fbs: int | None = None
    restecg: str
    exang: int
    slope: str
    ca: float | None = None
    thal: str | None = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(request: PredictionRequest) -> PredictionResponse:
    prediction, probability = predict(model, request.model_dump())

    return PredictionResponse(prediction=prediction, probability=probability)
