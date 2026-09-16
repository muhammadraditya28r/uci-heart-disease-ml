from sklearn.linear_model import LogisticRegression

from heart_disease.config import (
    RAW_DATA_DIR,
    MODEL_DIR,
    FEATURES,
    TARGET_COLUMN,
    ExperimentConfig,
)
from heart_disease.data.ingestion import load_file
from heart_disease.features.cleaning import clean_data
from heart_disease.features.preprocessing import (
    create_numeric_pipeline,
    create_categorical_pipeline,
    create_preprocessor,
    create_training_pipeline,
)
from heart_disease.models.train import save_model, train_model


config = ExperimentConfig()


def main() -> None:
    df = load_file(RAW_DATA_DIR / "heart_disease_uci.csv")
    df = clean_data(df, drop_thresh=config.drop_threshold)

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    preprocessor = create_preprocessor(
        create_numeric_pipeline(
            use_scaler=config.use_scaler, add_indicator=config.numeric_missing_indicator
        ),
        create_categorical_pipeline(add_indicator=config.categorical_missing_indicator),
    )
    model = create_training_pipeline(
        LogisticRegression(**config.logistic_regression_params), preprocessor
    )

    model = train_model(model, X, y)

    save_model(model, MODEL_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
