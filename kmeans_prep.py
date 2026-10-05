"""Helper code for the K-Means model (Model 6).
Kept in its own file so the saved joblib pipeline can be loaded later by predict.py / app.py."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# one-hot with drop_first=True (same result as the group pipeline's pd.get_dummies): 'Aug' and 'New_Visitor' are the dropped baselines
MONTHS_KEPT = ['Dec', 'Feb', 'Jul', 'June', 'Mar', 'May', 'Nov', 'Oct', 'Sep']
VISITORS_KEPT = ['Other', 'Returning_Visitor']


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row feature building (no statistics are learned here, so it is safe to run BEFORE the split).
    Mirrors the group pipeline: log1p columns, engineered totals, 0/1 encoding, one-hot encoding."""
    df = raw.copy()
    for c in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']:
        df[c + '_log'] = np.log1p(df[c])
    df['Total_Pages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
    df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['Avg_Duration_Per_Page'] = df['Total_Duration'] / (df['Total_Pages'] + 1.0)
    df['Weekend'] = df['Weekend'].astype(int)
    for m in MONTHS_KEPT:
        df['Month_' + m] = (df['Month'] == m).astype(int)
    for v in VISITORS_KEPT:
        df['VisitorType_' + v] = (df['VisitorType'] == v).astype(int)
    if 'Revenue' in df.columns:               # not present for a brand-new live session
        df['Revenue'] = df['Revenue'].astype(int)
    return df.drop(columns=['Month', 'VisitorType'])


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keeps only the chosen columns (so each variety can use a different feature set)."""
    def __init__(self, cols=None):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[list(self.cols)]


class Winsorizer(BaseEstimator, TransformerMixin):
    """IQR outlier capping. Bounds are learned on the training part only, then applied to any data."""
    def __init__(self, factor=1.5):
        self.factor = factor
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).values
        self.upper_ = (q3 + self.factor * iqr).values
        # columns where IQR == 0 (e.g. mostly zeros) would collapse to a constant -> leave them uncapped
        self.skip_ = (iqr.values == 0)
        return self
    def transform(self, X):
        A = np.asarray(X, dtype=float).copy()
        lo = np.where(self.skip_, -np.inf, self.lower_)
        hi = np.where(self.skip_, np.inf, self.upper_)
        return np.clip(A, lo, hi)


class ClusterFeaturizer(BaseEstimator, TransformerMixin):
    """Used only for the 'does a cluster label help a supervised model?' experiment.
    Fits scaler + K-Means on the training part only and appends one-hot cluster columns."""
    def __init__(self, cols=None, k=3):
        self.cols = cols
        self.k = k
    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        self.sc_ = StandardScaler().fit(X[list(self.cols)])
        self.km_ = KMeans(self.k, n_init=10, random_state=42).fit(self.sc_.transform(X[list(self.cols)]))
        return self
    def transform(self, X):
        lab = self.km_.predict(self.sc_.transform(X[list(self.cols)]))
        return np.hstack([X.values, np.eye(self.k)[lab]])
