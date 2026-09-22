from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def predict(model: Pipeline, features: dict[str, Any]) -> tuple[int, float]:
    """Generate a prediction and positive-class probability."""

    X = pd.DataFrame([features])

    prediction = int(model.predict(X)[0])

    probability = float(model.predict_proba(X)[0, 1])

    return prediction, probability
