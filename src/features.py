import pandas as pd

def prepare_features(df: pd.DataFrame):
    """
    Prepare features and target variable for modeling
    """
    # Drop high-cardinality IDs
    drop_cols = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "CustomerId",
        "SubscriptionId"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert datetime
    if "TransactionStartTime" in df.columns:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        df["transaction_hour"] = df["TransactionStartTime"].dt.hour
        df["transaction_day"] = df["TransactionStartTime"].dt.day
        df = df.drop(columns=["TransactionStartTime"])

    # Target variable
    y = df["FraudResult"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["FraudResult"])

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    return X, y
