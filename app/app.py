
# Load ../src modules by file path so imports work on Streamlit Cloud
import os, sys, importlib.util
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load(mod_name: str, rel_path: str):
    path = os.path.join(ROOT_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# src modules
dp  = _load("src.data_prep",          "src/data_prep.py")
fe  = _load("src.feature_engineering","src/feature_engineering.py")
cl  = _load("src.clustering",         "src/clustering.py")
ins = _load("src.insights",           "src/insights.py")

# re-exported names (so the rest of your code stays the same)
clean_transactions = dp.clean_transactions
infer_column_map   = dp.infer_column_map
apply_column_map   = dp.apply_column_map
REQUIRED_COLUMNS   = dp.REQUIRED_COLUMNS
OPTIONAL_COLUMNS   = dp.OPTIONAL_COLUMNS

build_customer_features = fe.build_customer_features

get_core_feature_columns   = cl.get_core_feature_columns
scale_features             = cl.scale_features
kmeans_with_silhouette     = cl.kmeans_with_silhouette
pca_project                = cl.pca_project

build_insights_table = ins.build_insights_table

# libs
import streamlit as st
import pandas as pd
import plotly.express as px

import io

# Cache reading of the uploaded file (fast re-runs)
@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    return pd.read_csv(io.BytesIO(file_bytes))

st.set_page_config(page_title="Customer Segmentation & Insights (Modular)", layout="wide")
st.title("Customer Segmentation & Insights Dashboard (Modular)")

# ----------------------------
# Sidebar: upload + options
# ----------------------------
with st.sidebar:
    st.header("1) Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    st.header("2) Clustering options")
    k_min = st.number_input("k min", 2, 10, 3, 1)
    k_max = st.number_input("k max", 2, 15, 7, 1)
    if k_max < k_min:
        st.error("k max must be ≥ k min")

if file is None:
    st.info(
        "Upload a transactions CSV/XLSX. We'll auto-detect column names.\n\n"
        "Canonical columns: InvoiceNo, InvoiceDate, Quantity, UnitPrice (+ CustomerID, Country, StockCode, Description)."
    )
    st.stop()

# ----------------------------
# Read file robustly
# ----------------------------
try:
    raw = read_uploaded_file(file.getbuffer(), file.name)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()


st.subheader("Raw preview")
st.dataframe(raw.head(10), use_container_width=True)

# ----------------------------
# Column mapping (auto + UI)
# ----------------------------
st.subheader("Column mapping")
proposed_map, scores = infer_column_map(raw)

with st.expander("Edit/confirm mapping", expanded=True):
    mapping = {}
    choices = ["— none —"] + list(raw.columns)
    # Required first, then optional
    for canon in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        default = proposed_map.get(canon)
        idx = choices.index(default) if default in choices else 0
        sel = st.selectbox(f"{canon}", options=choices, index=idx, key=f"map_{canon}")
        mapping[canon] = None if sel == "— none —" else sel

# Validate required selections
missing_required = [c for c in REQUIRED_COLUMNS if not mapping.get(c)]
if missing_required:
    st.error(f"Please map required columns: {', '.join(missing_required)}")
    st.stop()

# Apply mapping for preview
raw_mapped = apply_column_map(raw, {k: v for k, v in mapping.items() if v})
st.caption("Preview after applying mapping")
st.dataframe(raw_mapped.head(8), use_container_width=True)

# ----------------------------
# Clean + feature engineering (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def clean_and_featurize(raw_df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = clean_transactions(raw_df, column_map=mapping)
    feats = build_customer_features(tx)
    return tx, feats

with st.spinner("Cleaning & feature engineering..."):
    tx, feats = clean_and_featurize(
        raw_mapped,
        {k: v for k, v in mapping.items() if v}
    )

# ---- QC snapshot + downloads ----
st.subheader("Quality snapshot")
c1, c2, c3, c4 = st.columns(4)
raw_rows = len(raw_mapped)
clean_rows = len(tx)
drop_pct = (raw_rows - clean_rows) / max(raw_rows, 1) * 100
date_min = tx["InvoiceDate"].min() if "InvoiceDate" in tx.columns else None
date_max = tx["InvoiceDate"].max() if "InvoiceDate" in tx.columns else None
c1.metric("Rows (raw → clean)", f"{raw_rows:,} → {clean_rows:,}", f"-{drop_pct:.1f}%")
c2.metric("Customers", f"{feats.shape[0]:,}")
c3.metric("Date start", str(date_min.date()) if date_min is not None else "—")
c4.metric("Date end", str(date_max.date()) if date_max is not None else "—")

d1, d2 = st.columns(2)
with d1:
    st.download_button("⬇️ Cleaned transactions CSV",
        tx.to_csv(index=False).encode("utf-8"),
        file_name="transactions_clean.csv", mime="text/csv")
with d2:
    st.download_button("⬇️ Customer features CSV",
        feats.to_csv(index=False).encode("utf-8"),
        file_name="customers_features.csv", mime="text/csv")

# ---- Customer features preview + core features ----
st.subheader("Customer Features")
st.write(f"Shape: {feats.shape}")
st.dataframe(feats.head(10), use_container_width=True)

core_cols = get_core_feature_columns(feats)
if len(core_cols) < 3 or feats.shape[0] < max(10, 2 * k_min):
    st.warning("Dataset looks small or missing key features. Consider widening k-range or ensuring enough customers.")

# ---- Scale + cluster ----
X_scaled, _ = scale_features(feats, core_cols)
with st.spinner("Selecting k via silhouette & clustering..."):
    best = kmeans_with_silhouette(X_scaled, k_min=k_min, k_max=k_max)
    if best["k"] is None:
        st.error("Could not find a valid k. Try a different k range or check data quality.")
        st.stop()
    feats["Cluster"] = best["labels"]

st.success(f"Selected k = {best['k']}  |  silhouette = {best['score']:.3f}")


# ----------------------------
# Profiles, charts, insights
# ----------------------------
profile = feats.groupby("Cluster")[core_cols].mean().sort_index()
counts = feats["Cluster"].value_counts().sort_index()
profile["Count"] = counts.values

ins_df, name_map = build_insights_table(profile)

colA, colB = st.columns(2)
with colA:
    st.subheader("Cluster Sizes")
    counts_df = pd.DataFrame({"Cluster": counts.index, "Customers": counts.values})
    st.plotly_chart(px.bar(counts_df, x="Cluster", y="Customers"), use_container_width=True)

with colB:
    if "Monetary" in profile.columns:
        st.subheader("Avg Monetary by Cluster")
        prof_df = profile.reset_index().rename(columns={"index": "Cluster"})
        st.plotly_chart(
            px.bar(prof_df, x="Cluster", y="Monetary", labels={"Monetary": "Avg Monetary"}),
            use_container_width=True,
        )

# PCA scatter
pts2 = pca_project(X_scaled, n_components=2)
pts2["Cluster"] = feats["Cluster"].values
st.plotly_chart(
    px.scatter(pts2, x="PC1", y="PC2", color="Cluster", title="Clusters (PCA 2D Projection)", opacity=0.85),
    use_container_width=True,
)

# Distributions
st.subheader("RFM-style Distributions by Cluster")
for col in ["RecencyDays", "Frequency", "Monetary"]:
    if col in feats.columns:
        st.plotly_chart(px.box(feats, x="Cluster", y=col, points="outliers"), use_container_width=True)

# Insights table
st.subheader("Insights & Recommendations")
st.dataframe(ins_df, use_container_width=True)

# Download mapping
assign = feats[["CustomerID", "Cluster"]].copy()
assign["Label"] = assign["Cluster"].map(name_map)
st.download_button(
    "⬇️ Download customer→cluster CSV",
    assign.to_csv(index=False).encode("utf-8"),
    file_name="customer_clusters.csv",
    mime="text/csv",
)

st.caption("Tip: Use this CSV for CRM targeting, win-back campaigns, and lookalike audiences.")
