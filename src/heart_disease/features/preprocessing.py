from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin, TransformerMixin
from collections.abc import Sequence
from heart_disease.utils.logging import get_logger
from heart_disease.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = get_logger(__name__)


def create_numeric_pipeline(
    scaler: TransformerMixin | None = None,
    use_scaler: bool = True,
    strategy: str = "median",
    add_indicator: bool = True,
) -> Pipeline:
    """
    Imputing and scaling numeric values.

    Returns:
        Pipeline for numerical features
    """

    if use_scaler:
        if scaler is None:
            logger.debug(
                "Create numeric pipeline with SimpleImputer and StandardScaler"
            )
            return Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy=strategy, add_indicator=add_indicator),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            logger.debug("Create numeric pipeline with SimpleImputer and %s", scaler)
            return Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy=strategy, add_indicator=add_indicator),
                    ),
                    ("scaler", scaler),
                ]
            )
    else:
        logger.debug("Create numeric pipeline with SimpleImputer and without scaler")
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=strategy, add_indicator=add_indicator),
                )
            ]
        )


def create_categorical_pipeline(
    encoder: TransformerMixin | None = None,
    strategy: str = "most_frequent",
    add_indicator: bool = True,
) -> Pipeline:
    """
    Imputing and encoding categorical values.

    Returns:
        Pipeline for categorical features
    """
    if encoder is None:
        logger.debug("Create categorical pipeline with SimpleImputer and OneHotEncoder")
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=strategy, add_indicator=add_indicator),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
    else:
        logger.debug("Create categorical pipeline with SimpleImputer and %s", encoder)
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=strategy, add_indicator=add_indicator),
                ),
                ("encoder", encoder),
            ]
        )


def create_preprocessor(
    numeric_pipeline: Pipeline | None = None,
    categorical_pipeline: Pipeline | None = None,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for categorical and numerical features.

    Returns:
        ColumnTransformer for preprocessing numeric and categorical features.
    """

    if numeric_pipeline is None:
        numeric_pipeline = create_numeric_pipeline()
    if categorical_pipeline is None:
        categorical_pipeline = create_categorical_pipeline()

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def create_training_pipeline(
    model: ClassifierMixin,
    preprocessor: ColumnTransformer | None = None,
) -> Pipeline:
    """
    Create full pipeline for model training

    Returns:
        Pipeline
    """
    if preprocessor is None:
        preprocessor = create_preprocessor()

    logger.debug(
        "Create training pipeline with %s model and %s preprocessor",
        model,
        preprocessor,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
