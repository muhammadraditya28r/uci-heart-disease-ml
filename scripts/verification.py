import pandas as pd
from numpy import nan

from heart_disease.models.train import load_model
from heart_disease.config import MODEL_DIR


model = load_model(path=MODEL_DIR / "01_final_lr_model.joblib")


sample = pd.DataFrame(
    [
        {
            "age": 63,
            "trestbps": 145,
            "chol": 233,
            "thalch": 150,
            "oldpeak": 2.3,
            "sex": "Female",
            "cp": "typical angina",
            "fbs": nan,
            "restecg": "lv hypertrophy",
            "exang": 0,
            "slope": "downsloping",
            "ca": 0,
            "thal": "fixed defect",
        }
    ]
)

prediction = model.predict(sample)

print(prediction)

probability = model.predict_proba(sample)

print(probability)
