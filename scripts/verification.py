import pandas as pd
from numpy import nan

from heart_disease.models.train import load_model
from heart_disease.models.interpretation import get_feature_coefficients
from heart_disease.config import MODEL_DIR


model = load_model(path=MODEL_DIR / "01_final_lr_model.joblib")


print(model)


sample = pd.DataFrame(
    [
        {
            "age": 63,
            "trestbps": 145,
            "chol": 233,
            "thalch": 150,
            "oldpeak": 2.3,
            "sex": 1,
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

coeff = get_feature_coefficients(model)

print(coeff)
