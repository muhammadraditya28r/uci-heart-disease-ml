# Heart Disease Prediction — End-to-End Machine Learning

An end-to-end machine learning project for predicting the presence of heart disease from clinical measurements. The project covers the complete workflow from exploratory analysis and data cleaning to model evaluation, experiment tracking, automated testing, containerization, and REST API deployment.

> **Project status:** Deployed inference API
> **Primary task:** Binary classification
> **Primary evaluation metric:** F1-score
> **Production model:** Logistic Regression
> **Deployment:** Docker + Render

## Overview

This project uses the UCI Heart Disease dataset to develop a reproducible classification pipeline for predicting whether a patient belongs to the positive heart-disease class.

Rather than treating model training as an isolated notebook exercise, the project separates experimentation from reusable production code:

```text
Raw Data
   │
   ▼
Data Ingestion
   │
   ▼
Data Cleaning
   │
   ├── Invalid-value handling
   ├── Missing-data analysis
   ├── Feature selection
   └── Target transformation
   │
   ▼
Train / Test Split
   │
   ▼
Preprocessing Pipeline
   │
   ├── Numerical imputation
   ├── Standardization
   ├── Categorical imputation
   └── One-hot encoding
   │
   ▼
Model Training & Cross-Validation
   │
   ├── Model comparison
   ├── Hyperparameter tuning
   └── MLflow tracking
   │
   ▼
Held-out Test Evaluation
   │
   ▼
Serialized Scikit-learn Pipeline
   │
   ▼
FastAPI
   │
   ▼
Docker
   │
   ▼
Render
```

## Key Results

The final Logistic Regression model was evaluated on a held-out test set containing 184 observations.

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.837 |
| Precision | 0.805 |
| Recall    | 0.931 |
| F1-score  | 0.864 |

The confusion matrix was:

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |          59 |          23 |
| Actual 1 |           7 |          95 |

The model therefore achieved high recall for the positive class while producing more false positives than false negatives.

Because the project prioritizes F1 rather than accuracy alone, model selection considers the balance between precision and recall rather than optimizing for overall correctness only.

## Dataset

The project uses the combined UCI Heart Disease dataset.

The modeling dataset contains 13 predictive features:

### Numerical features

* `age`
* `trestbps` — resting blood pressure
* `chol` — serum cholesterol
* `thalch` — maximum heart rate achieved
* `oldpeak` — ST depression

### Categorical features

* `sex`
* `cp` — chest pain type
* `fbs` — fasting blood sugar
* `restecg` — resting ECG
* `exang` — exercise-induced angina
* `slope` — ST segment slope
* `ca` — number of major vessels
* `thal` — thalassemia

The original target is transformed into a binary target:

```text
0 → absence of heart disease
1 → presence of heart disease
```

## Data Cleaning

The cleaning pipeline handles several issues identified during exploratory analysis.

### Invalid measurements

Non-positive values in clinically meaningful numerical variables are treated as missing:

```text
chol
trestbps
oldpeak
```

These values are converted to `NaN` before preprocessing.

### Missing data

The dataset contains substantial missingness in several features. Instead of performing global manual imputation before model training, imputation is incorporated into the scikit-learn pipeline.

This prevents preprocessing statistics from leaking across the train/test boundary.

### Row filtering

Rows are filtered using a configurable minimum-observation threshold rather than applying arbitrary deletion rules.

The selected configuration is:

```text
DROP_THRESHOLD = 9
```

## Feature Engineering and Preprocessing

The project uses a `ColumnTransformer` containing separate numerical and categorical pipelines.

### Numerical pipeline

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler
```

### Categorical pipeline

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

The complete preprocessing and classifier are stored inside one scikit-learn `Pipeline`.

This is important for production inference because the API receives raw feature values and applies exactly the same transformations used during training.

```text
Raw API input
     ↓
Saved sklearn Pipeline
     ↓
Imputation
     ↓
Scaling / Encoding
     ↓
Logistic Regression
     ↓
Prediction + probability
```

## Model Development

Several classification algorithms were investigated during experimentation, including:

* Logistic Regression
* Support Vector Classifier
* Random Forest
* Gradient Boosting
* K-Nearest Neighbors
* XGBoost

Models are evaluated using stratified cross-validation with multiple metrics:

```text
Accuracy
Precision
Recall
F1
ROC AUC
```

The primary selection metric is F1-score.

The project also considers practical model-selection factors such as interpretability, computational cost, and inference complexity rather than selecting a model solely because it produces the highest numerical score.

## Final Model

The production artifact is a serialized scikit-learn pipeline:

```text
models/01_final_lr_model.joblib
```

The pipeline contains:

```text
ColumnTransformer
        +
LogisticRegression
```

The selected Logistic Regression configuration is:

```text
C = 1
solver = "saga"
l1_ratio = 0.5
max_iter = 5000
random_state = 42
```

Because preprocessing is included in the serialized pipeline, the production API does not need to reproduce the training transformations manually.

## Model Interpretability

Because Logistic Regression is the production model, its coefficients can be inspected directly.

The project converts coefficients into odds ratios to make the model's learned relationships easier to interpret.

For example:

```text
odds ratio = exp(coefficient)
```

The evaluation notebook also investigates:

* feature coefficients
* odds ratios
* confusion matrix
* classification report
* false positives
* false negatives
* prediction probabilities

This provides a more useful analysis than reporting a single aggregate metric.

## Experiment Tracking

MLflow is used to track model-development experiments.

Tracked information includes:

* experiment configuration
* model parameters
* cross-validation metrics
* model comparison results
* hyperparameter-search results
* final evaluation metrics
* serialized models

The project therefore separates:

```text
Experimentation
        ↓
Evaluation
        ↓
Model selection
        ↓
Production artifact
```

rather than relying exclusively on notebook output.

## Project Structure

```text
uci-heart-disease-ml/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   └── 01_final_lr_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_preprocessing_and_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── scripts/
│   ├── train.py
│   ├── verify_model.py
│   └── smoke_test.py
│
├── src/
│   └── heart_disease/
│       ├── data/
│       │   ├── ingestion.py
│       │   └── splitting.py
│       │
│       ├── features/
│       │   ├── cleaning.py
│       │   └── preprocessing.py
│       │
│       ├── models/
│       │   ├── evaluation.py
│       │   ├── interpretation.py
│       │   ├── predict.py
│       │   ├── tracking.py
│       │   └── train.py
│       │
│       ├── visualization/
│       │   └── plot.py
│       │
│       ├── config.py
│       └── api.py
│
├── tests/
│   ├── api/
│   ├── data/
│   ├── features/
│   └── models/
│
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Testing

The project includes automated tests for:

* data ingestion
* data splitting
* data cleaning
* preprocessing pipelines
* missing-value handling
* unseen categorical values
* model evaluation
* API health endpoint
* API prediction endpoint
* invalid API requests

The API tests verify both successful inference and validation failures.

For example:

```text
POST /predict
        ↓
valid request → 200
invalid request → 422
```

## Continuous Integration

GitHub Actions runs the project's quality checks automatically.

The CI pipeline includes:

```text
Checkout
   ↓
Python 3.12
   ↓
uv dependency installation
   ↓
Ruff
   ↓
pytest
   ↓
Docker image build
   ↓
Docker smoke test
```

This ensures that application changes are tested before they are deployed.

## REST API

The model is exposed through FastAPI.

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "age": 63,
  "trestbps": 145,
  "chol": 233,
  "thalch": 150,
  "oldpeak": 2.3,
  "sex": "Male",
  "cp": "typical angina",
  "fbs": null,
  "restecg": "lv hypertrophy",
  "exang": 0,
  "slope": "downsloping",
  "ca": 0,
  "thal": "fixed defect"
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.509109432471831
}
```

The API uses Pydantic models for request validation and response serialization.

It also includes HTTP middleware for request logging and execution-time measurement.

## Running Locally

### Requirements

* Python 3.12+
* `uv`
* Docker, if running the containerized API

### Install dependencies

```bash
uv sync
```

### Run tests

```bash
uv run pytest
```

### Run linting

```bash
uv run ruff check .
```

### Train the production model

```bash
uv run python scripts/train.py
```

### Run the API locally

```bash
uv run uvicorn heart_disease.api:app --host 0.0.0.0 --port 8000
```

The interactive API documentation is then available through FastAPI's generated OpenAPI interface.

## Running with Docker

Build the image:

```bash
docker build -t heart-disease-api .
```

Run the container:

```bash
docker run --rm \
  -p 8000:8000 \
  heart-disease-api
```

Check the health endpoint:

```bash
curl http://localhost:8000/health
```

The Docker image contains the runtime application, dependencies, source package, and serialized production model while excluding development-only files such as notebooks, datasets, tests, and reports.

The container runs the application as a non-root user.

## Deployment

The API is containerized using Docker and deployed as a web service on Render.

The production container:

```text
Dockerfile
    ↓
Python 3.12
    ↓
uv dependency environment
    ↓
FastAPI + Uvicorn
    ↓
Serialized sklearn pipeline
```

The application binds to:

```text
0.0.0.0:$PORT
```

allowing the deployment platform to provide the runtime port dynamically while retaining port `8000` as the local default.

## Design Principles

Several engineering decisions are intentional:

### Reproducible preprocessing

Preprocessing is part of the scikit-learn pipeline instead of being performed manually in notebooks.

### Separation of concerns

Notebook experimentation is separated from reusable source code.

```text
notebooks/
    experimentation

src/
    reusable application logic

scripts/
    executable workflows

tests/
    automated verification
```

### Configuration-driven experiments

Experiment parameters such as:

```text
random seed
test size
CV folds
scoring metric
missing-data threshold
preprocessing options
model hyperparameters
```

are centralized in the project configuration.

### Production parity

The same serialized pipeline used for inference is packaged into the Docker image and loaded by FastAPI.

This reduces the risk of training/inference preprocessing inconsistencies.

### Automated verification

The project tests both the Python application and the containerized application.

## Limitations

This project is intended as a machine learning engineering and portfolio project, not as a clinical diagnostic system.

Important limitations include:

* The dataset is relatively small by modern machine-learning standards.
* The source data combines observations from multiple datasets/institutions.
* Missing data is substantial for several variables.
* Dataset characteristics may not represent a modern clinical population.
* Model probabilities should not be interpreted as clinically calibrated risk without additional validation.
* The model has not been externally validated on an independent clinical dataset.
* The API should not be used to make medical decisions.

The model demonstrates an ML workflow and deployment architecture; it does not constitute medical advice or a validated clinical decision-support system.

## What This Project Demonstrates

This project is intended to demonstrate practical data-science and machine-learning engineering capabilities across the full lifecycle:

```text
Data Understanding
        ↓
Data Quality
        ↓
Exploratory Analysis
        ↓
Feature Preprocessing
        ↓
Model Development
        ↓
Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Interpretability
        ↓
Experiment Tracking
        ↓
Automated Testing
        ↓
CI
        ↓
Docker
        ↓
FastAPI
        ↓
Cloud Deployment
```

The emphasis is not only on obtaining a model score, but on building a reproducible path from raw data to a deployable inference service.
