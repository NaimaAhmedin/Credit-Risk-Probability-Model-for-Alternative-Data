import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
print("📥 Loading processed data...")
df = pd.read_csv(DATA_DIR / "data_with_proxy_target.csv")
y = df["is_high_risk"]

# Drop high-cardinality IDs
drop_cols = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "CustomerId",
    "SubscriptionId"
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["is_high_risk"])

print(f"Shape X: {X.shape}, Positive rate: {y.mean():.4f}")

# ----------------------------
# Identify column types
# ----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# ----------------------------
# Preprocessing pipeline
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
    ]
)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment("Credit_Risk_Model_Experiment")

def evaluate_model(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics

def train_and_log_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    with mlflow.start_run(run_name=name):
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]
        metrics = evaluate_model(y_te, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path=name, registered_model_name=name)

        print(f"\n📊 {name} Results")
        print(metrics)
        print(classification_report(y_te, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
        
        return pipeline, metrics

# ----------------------------
# Logistic Regression
# ----------------------------
lr_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga"
    ))
])

lr_model, lr_metrics = train_and_log_model(
    "LogisticRegression_CreditRisk",
    lr_pipeline,
    X_train, y_train,
    X_test, y_test
)

# ----------------------------
# Random Forest (numerical only)
# ----------------------------
rf_pipeline = Pipeline([
    ("preprocess", ColumnTransformer([("num", StandardScaler(), numerical_cols)])),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))
])

rf_model, rf_metrics = train_and_log_model(
    "RandomForest_CreditRisk",
    rf_pipeline,
    X_train, y_train,
    X_test, y_test
)

# ----------------------------
# Save best model locally
# ----------------------------
best_model_name = "LogisticRegression_CreditRisk" if lr_metrics["roc_auc"] >= rf_metrics["roc_auc"] else "RandomForest_CreditRisk"
best_model = lr_model if best_model_name.startswith("Logistic") else rf_model
joblib.dump(best_model, MODEL_DIR / f"{best_model_name}_best.pkl")
print(f"\n✅ Best model saved: {best_model_name}_best.pkl")
