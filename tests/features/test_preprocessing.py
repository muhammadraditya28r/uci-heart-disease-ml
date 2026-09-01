from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from heart_disease.features.preprocessing import (
    create_categorical_pipeline,
    create_numeric_pipeline,
    create_preprocessor,
    create_training_pipeline,
)


def test_create_numeric_pipeline() -> None:
    pipeline = create_numeric_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "imputer" in pipeline.named_steps
    assert "scaler" in pipeline.named_steps


def test_create_categorical_pipeline() -> None:
    pipeline = create_categorical_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "imputer" in pipeline.named_steps
    assert "encoder" in pipeline.named_steps


def test_create_preprocessor() -> None:
    preprocessor = create_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)

    transformer_names = [
        name for name, _, _ in preprocessor.transformers
    ]

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


def test_create_training_pipeline() -> None:
    model = LogisticRegression()

    pipeline = create_training_pipeline(model=model)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"] is model