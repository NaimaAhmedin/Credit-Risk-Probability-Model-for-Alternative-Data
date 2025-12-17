import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA = BASE_DIR / "data" / "raw" / "data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

OUTPUT_FILE = PROCESSED_DIR / "data_with_proxy_target.csv"


def load_data():
    df = pd.read_csv(RAW_DATA)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def compute_rfm(df):
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


def cluster_customers(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm


def assign_high_risk(rfm):
    cluster_summary = (
        rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    # Least engaged cluster = low frequency & low monetary
    high_risk_cluster = (
        cluster_summary
        .sort_values(by=["Frequency", "Monetary"], ascending=True)
        .index[0]
    )

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]


def main():
    df = load_data()
    rfm = compute_rfm(df)
    rfm = cluster_customers(rfm)
    risk_labels = assign_high_risk(rfm)

    df_final = df.merge(risk_labels, on="CustomerId", how="left")

    df_final.to_csv(OUTPUT_FILE, index=False)

    print("✅ Task 4 completed successfully")
    print("High-risk distribution:")
    print(df_final["is_high_risk"].value_counts())


if __name__ == "__main__":
    main()
