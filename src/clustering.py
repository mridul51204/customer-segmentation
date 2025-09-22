from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CORE_FEATURES = ["RecencyDays","Frequency","Monetary","TenureDays","AvgBasketQty","UniqueProducts"]

def get_core_feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in CORE_FEATURES if c in feats.columns]

def scale_features(feats: pd.DataFrame, cols: list[str]):
    X = feats[cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def kmeans_with_silhouette(X_scaled: np.ndarray, k_min: int = 3, k_max: int = 6, random_state: int = 42):
    best = {"k": None, "score": -1, "labels": None, "model": None}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels)
        if score > best["score"]:
            best = {"k": k, "score": score, "labels": labels, "model": km}
    return best

def pca_project(X_scaled: np.ndarray, n_components: int = 2, random_state: int = 42) -> pd.DataFrame:
    pca = PCA(n_components=n_components, random_state=random_state)
    pts = pca.fit_transform(X_scaled)
    return pd.DataFrame(pts, columns=[f"PC{i+1}" for i in range(n_components)])
