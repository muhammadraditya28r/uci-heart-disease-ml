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
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def get_prediction_errors(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Return test observation with predictions and error classification."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    result = X_test.copy()
    result["actual"] = y_test.to_numpy()
    result["predicted"] = predictions
    result["probability"] = probabilities

    result["error_type"] = "correct"
    result.loc[
        (result["actual"] == 0) & (result["predicted"] == 1),
        "error_type",
    ] = "false_positive"
    result.loc[
        (result["actual"] == 1) & (result["predicted"] == 0),
        "error_type",
    ] = "false_negative"

    return result
