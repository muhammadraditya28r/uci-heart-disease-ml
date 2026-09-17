import math

import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return Logistic Regression coefficients with transformed feature names."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = classifier.coef_[0]

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "odds_ratio": [math.exp(coef) for coef in coefficients],
            }
        )
        .sort_values("coefficient")
        .reset_index(drop=True)
    )
