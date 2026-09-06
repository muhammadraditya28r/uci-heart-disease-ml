import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

    transformer_names = [name for name, _, _ in preprocessor.transformers]

    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


def test_create_training_pipeline() -> None:
    model = LogisticRegression()

    pipeline = create_training_pipeline(model=model)

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert pipeline.named_steps["classifier"] is model


@pytest.fixture
def test_training_pipeline_handles_missing_values(
    sample_features, sample_target
) -> None:
    pipeline = create_training_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(sample_features, sample_target)

    predictions = pipeline.predict(sample_features)

    assert len(predictions) == len(sample_target)


@pytest.fixture
def test_traiining_pipeline_handles_unseen_categories(
    sample_features, sample_target
) -> None:
    X_test = sample_features.copy()
    X_test.loc[0, "cp"] = 999

    pipeline = create_training_pipeline(LogisticRegression(max_iter=1000))

    pipeline.fit(sample_features, sample_target)

    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(X_test)


@pytest.fixture
def test_training_pipeline_contains_scaler(sample_features, sample_target):
    pipeline = create_training_pipeline(LogisticRegression())
    pipeline.fit(sample_features, sample_target)
    assert (
        "scaler"
        in pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps
    )


@pytest.fixture
def test_training_pipeline_does_not_contains_scaler(sample_features, sample_target):
    pipeline = create_training_pipeline(RandomForestClassifier())
    pipeline.fit(sample_features, sample_target)
    assert (
        "scaler"
        not in pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps
    )
