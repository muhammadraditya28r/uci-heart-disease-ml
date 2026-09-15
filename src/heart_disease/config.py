from typing import Any

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass, field


# -------------------------------------PROJECT FOLDER-----------------------------------------#

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"


# -----------------------------------------DATA COLUMNS-----------------------------------------------#

TARGET_COLUMN = "target"
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FULL_COLUMNS = FEATURES + [TARGET_COLUMN]


# -----------------------------------TRAINING CONFIGURATION-----------------------------------------#

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "uci-heart-disease"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
STRATIFIED_K_FOLDS = StratifiedKFold(
    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
)
SINGLE_SCORING = "f1"
MULTIPLE_SCORING = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


# --------------------------------SELECTED BY EXPERIMENTATIONS-----------------------------------------#

DROP_THRESHOLD = 9
USE_SCALER = True
NUMERIC_MISSING_INDICATOR = False
CATEGORICAL_MISSING_INDICATOR = True
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1,
    "solver": "saga",
    "l1_ratio": 0.5,
    "max_iter": 5000,
    "random_state": RANDOM_STATE,
}


# ------------------------------------EXPERIMENTATION CONFIG--------------------------------------------#


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    single_scoring: str = SINGLE_SCORING
    multiple_scoring: list[str] = field(default_factory=lambda: MULTIPLE_SCORING.copy())
    drop_threshold: int = DROP_THRESHOLD
    use_scaler: bool = USE_SCALER
    numeric_missing_indicator: bool = NUMERIC_MISSING_INDICATOR
    categorical_missing_indicator: bool = CATEGORICAL_MISSING_INDICATOR

    logistic_regression_params: dict[str, Any] = field(
        default_factory=lambda: LOGISTIC_REGRESSION_PARAMS.copy()
    )
