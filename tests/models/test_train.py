import pandas as pd

from sklearn.linear_model import LogisticRegression

from heart_disease.features.preprocessing import create_training_pipeline
from heart_disease.models.train import train_model


def test_train_model() -> None:
    X = pd.DataFrame(
        {
            "age": [40, 50, 60, 70],
            "trestbps": [120, 130, 140, 150],
            "chol": [200, 220, 240, 260],
            "thalch": [160, 150, 140, 130],
            "oldpeak": [0.0, 1.0, 1.5, 2.0],
            "sex": [1, 0, 1, 0],
            "cp": [1, 2, 1, 3],
            "fbs": [0, 0, 1, 0],
            "restecg": [0, 1, 0, 1],
            "exang": [0, 0, 1, 1],
            "slope": [1, 2, 1, 2],
            "ca": [0, 0, 1, 1],
            "thal": [3, 3, 7, 7],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    model = create_training_pipeline(model=LogisticRegression())

    result = train_model(model, X, y)

    assert result is not None
