from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"

RANDOM_STATE = 42
TARGET_COLUMN = 'target'

NUMERIC_FEATURES = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
)
CATEGORICAL_FEATURES = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
