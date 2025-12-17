import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path
import joblib

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# ----------------------------
# Load sample data
# ----------------------------
df = pd.read_csv(DATA_DIR / "data_with_proxy_target.csv")
y = df["is_high_risk"]

drop_cols = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "CustomerId",
    "SubscriptionId"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["is_high_risk"])
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ----------------------------
# Preprocessor
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
    ]
)

# ----------------------------
# Pipelines
# ----------------------------
lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced", solver="saga"))
])

rf_pipeline = Pipeline([
    ("preprocess", ColumnTransformer([("num", StandardScaler(), numerical_cols)])),
    ("model", RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", random_state=42))
])

# ----------------------------
# Tests
# ----------------------------
def test_preprocessor_output_shape():
    """Check that the preprocessor transforms data correctly"""
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] >= len(numerical_cols), "Preprocessor output columns too few"

def test_lr_pipeline_predict():
    """Test Logistic Regression pipeline predictions"""
    lr_pipeline.fit(X, y)
    y_pred = lr_pipeline.predict(X)
    assert len(y_pred) == len(y), "Predictions length mismatch"
    assert set(y_pred).issubset({0,1}), "Predictions should be binary"

def test_rf_pipeline_predict():
    """Test Random Forest pipeline predictions"""
    rf_pipeline.fit(X[numerical_cols], y)
    y_pred = rf_pipeline.predict(X[numerical_cols])
    assert len(y_pred) == len(y), "Predictions length mismatch"
    assert set(y_pred).issubset({0,1}), "Predictions should be binary"

def test_saved_model_exists():
    """Check that the best model file exists"""
    best_model_path = MODEL_DIR / "RandomForest_CreditRisk_best.pkl"
    assert best_model_path.exists(), "Best model file not found"
