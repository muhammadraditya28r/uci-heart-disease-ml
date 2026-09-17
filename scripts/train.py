from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin

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
from heart_disease.models.tracking import log_model


config = ExperimentConfig()


def build_model(config: ExperimentConfig, classifier: ClassifierMixin) -> Pipeline:
    """Build the selected preprocessing + model pipeline"""

    preprocessor = create_preprocessor(
        create_numeric_pipeline(
            use_scaler=config.use_scaler,
            add_indicator=config.numeric_missing_indicator,
        ),
        create_categorical_pipeline(
            add_indicator=config.categorical_missing_indicator,
        ),
    )

    return create_training_pipeline(
        classifier,
        preprocessor,
    )


def main() -> None:

    df = load_file(RAW_DATA_DIR / "heart_disease_uci.csv")
    df = clean_data(df, drop_thresh=config.drop_threshold)

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    model = build_model(
        config=config,
        classifier=LogisticRegression(**config.logistic_regression_params),
    )

    model = train_model(model, X, y)

    save_model(model, MODEL_DIR / "01_final_lr_model.joblib")

    log_model(model, name="production_model")


if __name__ == "__main__":
    main()
