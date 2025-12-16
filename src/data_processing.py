import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from features import (
    AggregateTransactionFeatures,
    DateTimeFeatures,
    WoETransformer
)

# -------------------------------
# Load Data
# -------------------------------
DATA_PATH = "data/raw/data.csv"
OUTPUT_PATH = "data/processed/features.csv"
TARGET = "FraudResult"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

# -------------------------------
# Pipelines
# -------------------------------
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# -------------------------------
# Full Feature Engineering Pipeline
# -------------------------------
pipeline = Pipeline(steps=[
    ("aggregates", AggregateTransactionFeatures()),
    ("datetime", DateTimeFeatures()),
    ("preprocessing", preprocessor),
])

from scipy import sparse

X_processed = pipeline.fit_transform(X)

# Save sparse matrix correctly
sparse.save_npz("data/processed/features_sparse.npz", X_processed)

# Save target separately
y.to_csv("data/processed/target.csv", index=False)

print("✅ Feature engineering completed. Sparse features saved.")
