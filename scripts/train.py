from sklearn.linear_model import LogisticRegression

from heart_disease.config import (
    RAW_DATA_DIR,
    RANDOM_STATE,
    MODEL_DIR,
    FEATURES,
    TARGET_COLUMN,
)
from heart_disease.data.ingestion import load_file
from heart_disease.features.cleaning import clean_data
from heart_disease.features.preprocessing import create_training_pipeline
from heart_disease.models.train import save_model


def main() -> None:
    df = load_file(RAW_DATA_DIR / "heart_disease_uci.csv")
    df = clean_data(df, drop_thresh=10)

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    model = create_training_pipeline(
        LogisticRegression(
            max_iter=5000, l1_ratio=0.5, C=1, solver="saga", random_state=RANDOM_STATE
        )
    )

    model.fit(X, y)

    save_model(model, MODEL_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
