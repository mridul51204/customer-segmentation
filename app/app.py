# app/app.py (top)
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_prep import (
    clean_transactions, infer_column_map, apply_column_map,
    REQUIRED_COLUMNS, OPTIONAL_COLUMNS
)
from src.feature_engineering import build_customer_features
from src.clustering import (
    get_core_feature_columns, scale_features,
    kmeans_with_silhouette, pca_project
)
from src.insights import build_insights_table
from src.clustering import (
    get_core_feature_columns,
    scale_features,
    kmeans_with_silhouette,
    pca_project,
)
from src.insights import build_insights_table

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
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file, engine="openpyxl")
    else:
        raw = pd.read_csv(file)
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
# Clean + feature engineering
# ----------------------------
with st.spinner("Cleaning & feature engineering..."):
    tx = clean_transactions(raw, column_map={k: v for k, v in mapping.items() if v})
    feats = build_customer_features(tx)

st.subheader("Customer Features")
st.write(f"Shape: {feats.shape}")
st.dataframe(feats.head(10), use_container_width=True)

core_cols = get_core_feature_columns(feats)
if len(core_cols) < 3 or feats.shape[0] < max(10, 2 * k_min):
    st.warning(
        "Dataset looks small or missing key features. Consider widening k-range or ensuring enough customers."
    )

# ----------------------------
# Scale + cluster (auto k by silhouette)
# ----------------------------
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
