import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Aggregate Features per Customer
# -------------------------------
class AggregateTransactionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = (
            X.groupby("CustomerId")
            .agg(
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                transaction_count=("Amount", "count"),
                std_amount=("Amount", "std"),
            )
            .reset_index()
        )

        X = X.merge(agg, on="CustomerId", how="left")
        return X


# -------------------------------
# Datetime Feature Extraction
# -------------------------------
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])

        X["transaction_hour"] = X[self.datetime_col].dt.hour
        X["transaction_day"] = X[self.datetime_col].dt.day
        X["transaction_month"] = X[self.datetime_col].dt.month
        X["transaction_year"] = X[self.datetime_col].dt.year

        return X


# -------------------------------
# Weight of Evidence (WoE)
# -------------------------------
def calculate_woe_iv(df, feature, target):
    eps = 1e-6
    grouped = df.groupby(feature)[target]

    dist_good = grouped.apply(lambda x: (x == 0).sum())
    dist_bad = grouped.apply(lambda x: (x == 1).sum())

    dist_good = dist_good / dist_good.sum()
    dist_bad = dist_bad / dist_bad.sum()

    woe = np.log((dist_good + eps) / (dist_bad + eps))
    iv = ((dist_good - dist_bad) * woe).sum()

    return woe.to_dict(), iv


class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, target):
        self.categorical_cols = categorical_cols
        self.target = target
        self.woe_maps = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target] = y

        for col in self.categorical_cols:
            woe_map, iv = calculate_woe_iv(df, col, self.target)
            self.woe_maps[col] = woe_map

        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps.items():
            X[col] = X[col].map(woe_map).fillna(0)
        return X