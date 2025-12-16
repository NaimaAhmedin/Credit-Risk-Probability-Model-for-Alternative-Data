import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.features import prepare_features

# Load data
df = pd.read_csv("data/raw/data.csv")

# Feature engineering
X, y = prepare_features(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

best_auc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    print(f"\n{name}")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", auc)

    if auc > best_auc:
        best_auc = auc
        best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")
