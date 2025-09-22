# app/app.py (very top)
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# now safe to import
from src.data_prep import clean_transactions
from src.feature_engineering import build_customer_features
from src.clustering import (
    get_core_feature_columns, scale_features, kmeans_with_silhouette, pca_project
)
from src.insights import build_insights_table

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_prep import clean_transactions
from src.feature_engineering import build_customer_features
from src.clustering import (
    get_core_feature_columns, scale_features, kmeans_with_silhouette, pca_project
)
from src.insights import build_insights_table

st.set_page_config(page_title="Customer Segmentation & Insights", layout="wide")

st.title("Customer Segmentation & Insights Dashboard (Modular)")

with st.sidebar:
    st.header("1) Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    st.header("2) Clustering options")
    k_min = st.number_input("k min", 2, 10, 3, 1)
    k_max = st.number_input("k max", 2, 15, 6, 1)
    if k_max < k_min:
        st.error("k max must be ≥ k min")

if file is None:
    st.info("Upload a CSV/XLSX with: InvoiceNo, Quantity, UnitPrice, InvoiceDate; (CustomerID/Country optional).")
    st.stop()

try:
    raw = pd.read_excel(file) if file.name.lower().endswith((".xlsx",".xls")) else pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.subheader("Raw preview")
st.dataframe(raw.head(10), use_container_width=True)

with st.spinner("Cleaning & feature engineering..."):
    tx = clean_transactions(raw)
    feats = build_customer_features(tx)

st.subheader("Customer Features")
st.write(f"Shape: {feats.shape}")
st.dataframe(feats.head(10), use_container_width=True)

core_cols = get_core_feature_columns(feats)
X_scaled, _ = scale_features(feats, core_cols)

with st.spinner("Selecting k via silhouette & clustering..."):
    best = kmeans_with_silhouette(X_scaled, k_min=k_min, k_max=k_max)
    if best["k"] is None:
        st.error("Could not find a valid k. Try another k-range or check data quality.")
        st.stop()
    labels = best["labels"]
    feats["Cluster"] = labels

st.success(f"Selected k = {best['k']}  |  silhouette = {best['score']:.3f}")

profile = (feats.groupby("Cluster")[core_cols].mean().sort_index())
counts = feats["Cluster"].value_counts().sort_index()
profile["Count"] = counts.values

ins_df, name_map = build_insights_table(profile)

colA, colB = st.columns(2)
with colA:
    st.subheader("Cluster Sizes")
    st.plotly_chart(px.bar(counts, labels={"value":"Customers","index":"Cluster"}), use_container_width=True)
with colB:
    if "Monetary" in profile.columns:
        st.subheader("Avg Monetary by Cluster")
        st.plotly_chart(px.bar(profile, x=profile.index, y="Monetary",
                               labels={"x":"Cluster","Monetary":"Avg Monetary"}), use_container_width=True)

# PCA scatter
pts2 = pca_project(X_scaled, n_components=2)
pts2["Cluster"] = labels
st.plotly_chart(px.scatter(pts2, x="PC1", y="PC2", color="Cluster",
                           title="Clusters (PCA 2D Projection)", opacity=0.8),
                use_container_width=True)

st.subheader("Insights & Recommendations")
st.dataframe(ins_df, use_container_width=True)

# Download mapping
assign = feats[["CustomerID","Cluster"]].copy()
assign["Label"] = assign["Cluster"].map(name_map)
st.download_button(
    "⬇️ Download customer→cluster CSV",
    assign.to_csv(index=False).encode("utf-8"),
    file_name="customer_clusters.csv",
    mime="text/csv"
)
