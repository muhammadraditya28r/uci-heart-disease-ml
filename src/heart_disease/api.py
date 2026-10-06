import time

from fastapi import FastAPI, Request
from starlette.responses import Response
from pydantic import BaseModel

from heart_disease.utils.logging import get_logger
from heart_disease.config import PRODUCTION_MODEL_DIR
from heart_disease.models.predict import predict
from heart_disease.models.train import load_model


logger = get_logger(__name__)


app = FastAPI(
    title="Heart Disease Prediction API",
    version="0.1.0",
    description="API for predicting heart disease from clinical feature.",
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


@app.middleware("http")
async def log_request(request: Request, call_next) -> Response:
    start = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request completed | method=%s | path=%s | status=%s | duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "request failed | method=%s | path=%s | duration_ms=%.2f",
            request.method,
            request.url.path,
            duration_ms
        )

        raise


