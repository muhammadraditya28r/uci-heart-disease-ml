import pandas as pd
from heart_disease.data.splitting import split_data


def test_split_data() -> None:
    df = pd.DataFrame(
        {
            "age": range(100),
            "chol": range(100, 200),
            "target": [0, 1] * 50,
        }
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="target",
        features_column=["age", "chol"],
        test_size=0.2,
        random_state=42,
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20



def test_split_data_does_not_include_target() -> None:
    df = pd.DataFrame(
        {
            "age": range(20),
            "chol": range(20, 40),
            "target": [0, 1] * 10,
        }
    )

    X_train, X_test, _, _ = split_data(
        df,
        target_column="target",
        features_column=["age", "chol"],
    )

    assert "target" not in X_train.columns
    assert "target" not in X_test.columns