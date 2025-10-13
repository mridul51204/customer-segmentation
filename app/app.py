# app.py — Single-file Customer Segmentation & Insights (modular API inlined)
# Keeps your existing UI/flow but removes dependency on src/* files.
from __future__ import annotations
import io
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42

# ----------------------------
# Canonical schema (as in your UI text)
# ----------------------------
REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
OPTIONAL_COLUMNS = ["CustomerID", "Country", "StockCode", "Description"]

# ----------------------------
# Helpers
# ----------------------------
def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _nonempty(x) -> bool:
    return x is not None and x != "" and not (isinstance(x, float) and math.isnan(x))

# ----------------------------
# Column inference + mapping
# ----------------------------
def infer_column_map(df: pd.DataFrame) -> tuple[Dict[str, Optional[str]], Dict[str, float]]:
    """
    Heuristically infer the best source column for each canonical field.
    Returns (mapping, scores) where scores are simple 0..1 confidences.
    """
    cols = list(df.columns)
    low = [c.lower() for c in cols]
    def pick(candidates: List[str]) -> tuple[Optional[str], float]:
        # exact
        for cand in candidates:
            if cand in low:
                return cols[low.index(cand)], 1.0
        # contains / fuzzy
        for i, l in enumerate(low):
            if any(c in l for c in candidates):
                return cols[i], 0.6
        return None, 0.0

    mapping, scores = {}, {}
    # required
    inv, s1 = pick(["invoiceno", "invoice_no", "orderid", "order_no", "billno", "invoice"])
    dt, s2  = pick(["invoicedate", "invoice_date", "date", "orderdate", "timestamp", "ts"])
    qty, s3 = pick(["quantity", "qty", "units"])
    up, s4  = pick(["unitprice", "unit_price", "price", "amount", "value"])
    # optional
    cid, s5 = pick(["customerid", "customer_id", "custid", "user_id", "clientid"])
    ctry, s6= pick(["country", "region", "market"])
    sku, s7 = pick(["stockcode", "stock_code", "sku", "productid", "product_id"])
    desc, s8= pick(["description", "productname", "product_name", "item"])

    for canon, val, sc in [
        ("InvoiceNo", inv, s1), ("InvoiceDate", dt, s2),
        ("Quantity", qty, s3), ("UnitPrice", up, s4),
        ("CustomerID", cid, s5), ("Country", ctry, s6),
        ("StockCode", sku, s7), ("Description", desc, s8),
    ]:
        mapping[canon] = val
        scores[canon] = sc
    return mapping, scores

def apply_column_map(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Returns a copy with selected columns renamed to canonical names.
    """
    ren = {src: canon for canon, src in mapping.items() if _nonempty(src) and src in df.columns}
    out = df[list(ren.keys())].copy()
    out = out.rename(columns=ren)
    return out

# ----------------------------
# Cleaning
# ----------------------------
def clean_transactions(raw_df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """
    Produces a tidy transactions table with canonical columns and types.
    Adds TransactionAmount = Quantity * UnitPrice.
    Filters bad rows, coerces types, and keeps only mapped columns.
    """
    df = apply_column_map(raw_df, column_map).copy()

    # Type coercions
    if "InvoiceDate" in df: df["InvoiceDate"] = _coerce_datetime(df["InvoiceDate"])
    if "Quantity" in df: df["Quantity"] = _num(df["Quantity"])
    if "UnitPrice" in df: df["UnitPrice"] = _num(df["UnitPrice"])

    # Basic quality filters
    if "InvoiceDate" in df:
        df = df[~df["InvoiceDate"].isna()]
    if "Quantity" in df:
        df = df[~df["Quantity"].isna()]
    if "UnitPrice" in df:
        df = df[~df["UnitPrice"].isna()]

    # Compute amount
    if all(c in df.columns for c in ["Quantity", "UnitPrice"]):
        df["TransactionAmount"] = df["Quantity"] * df["UnitPrice"]

    # Normalize IDs as strings (optional)
    if "CustomerID" in df:
        df["CustomerID"] = df["CustomerID"].astype(str).str.strip()

    return df

# ----------------------------
# Feature engineering (RFM)
# ----------------------------
def build_customer_features(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Builds RFM-style features per CustomerID:
      - RecencyDays: days since last purchase (relative to max InvoiceDate)
      - Frequency:   number of invoices
      - Monetary:    sum of TransactionAmount
    Additional columns are passed through for analysis convenience.
    """
    if "CustomerID" not in tx.columns:
        # create a pseudo ID if absent (keeps app usable)
        tx = tx.copy()
        tx["CustomerID"] = "Unknown"
    if tx.empty:
        return pd.DataFrame(columns=["CustomerID","RecencyDays","Frequency","Monetary"])

    tx = tx.copy()
    max_date = tx["InvoiceDate"].max() if "InvoiceDate" in tx.columns else None
    grp = tx.groupby("CustomerID", as_index=False)

    # Frequency
    if "InvoiceNo" in tx.columns:
        freq = grp["InvoiceNo"].nunique().rename(columns={"InvoiceNo":"Frequency"})
    else:
        freq = grp.size().rename(columns={"size":"Frequency"})

    # Monetary
    if "TransactionAmount" in tx.columns:
        mon = grp["TransactionAmount"].sum().rename(columns={"TransactionAmount":"Monetary"})
    else:
        # fallback: UnitPrice if present
        if "UnitPrice" in tx.columns:
            mon = grp["UnitPrice"].sum().rename(columns={"UnitPrice":"Monetary"})
        else:
            mon = freq.copy()
            mon["Monetary"] = 0.0
            mon = mon[["CustomerID","Monetary"]]

    # Recency
    if max_date is not None and "InvoiceDate" in tx.columns:
        last_dt = grp["InvoiceDate"].max().rename(columns={"InvoiceDate":"LastDate"})
        base = freq.merge(mon, on="CustomerID").merge(last_dt, on="CustomerID")
        base["RecencyDays"] = (max_date - base["LastDate"]).dt.days.astype(float)
        base = base.drop(columns=["LastDate"])
    else:
        base = freq.merge(mon, on="CustomerID")
        base["RecencyDays"] = np.nan

    # Basic hygiene
    for c in ["Frequency", "Monetary"]:
        base[c] = base[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Order columns
    cols = ["CustomerID", "RecencyDays", "Frequency", "Monetary"]
    return base[cols]

# ----------------------------
# Clustering utilities
# ----------------------------
def get_core_feature_columns(feats: pd.DataFrame) -> List[str]:
    core = [c for c in ["RecencyDays","Frequency","Monetary"] if c in feats.columns]
    return core

def scale_features(feats: pd.DataFrame, cols: List[str]) -> tuple[pd.DataFrame, StandardScaler]:
    X = feats[cols].copy()
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X.values), columns=cols, index=feats.index)
    return Xs, scaler

def kmeans_with_silhouette(
    X: pd.DataFrame, k_min: int = 2, k_max: int = 10
) -> dict:
    best = {"k": None, "score": -1, "labels": None}
    # Guard small datasets
    k_max_eff = min(k_max, max(k_min, len(X) - 1))
    for k in range(k_min, k_max_eff + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        labels = model.fit_predict(X)
        # silhouette may fail if all points land in one label; guard
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = -1
        if score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels}
    return best

def pca_project(X: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    p = PCA(n_components=n_components, random_state=RANDOM_STATE)
    Z = p.fit_transform(X.values)
    cols = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(Z, columns=cols, index=X.index)

# ----------------------------
# Insights
# ----------------------------
def build_insights_table(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[int, str]]:
    """
    Given per-cluster means of core features, produce tags + actionable notes.
    Returns (insights_table, label_map).
    """
    df = profile_df.copy()
    # Heuristic labels
    labels = {}
    for c, row in df.iterrows():
        r = row.get("RecencyDays", np.nan)
        f = row.get("Frequency", 0.0)
        m = row.get("Monetary", 0.0)
        if pd.notna(r):
            if m >= df["Monetary"].quantile(0.75) and r <= df["RecencyDays"].quantile(0.4):
                name = "💎 Loyal High-Value"
            elif r >= df["RecencyDays"].quantile(0.8):
                name = "🧊 At-Risk / Dormant"
            elif f >= df["Frequency"].quantile(0.7) and m < df["Monetary"].quantile(0.4):
                name = "🧱 Frequent Low-Ticket"
            else:
                name = "✨ General"
        else:
            name = "✨ General"
        labels[int(c)] = name

    out = []
    for c, row in df.iterrows():
        name = labels[int(c)]
        idea = ""
        if "High-Value" in name:
            idea = "VIP perks, tiered pricing, early access, referral nudges."
        elif "At-Risk" in name:
            idea = "Win-back: reminder + time-bound coupon; highlight new arrivals."
        elif "Low-Ticket" in name:
            idea = "Bundle offers, free shipping thresholds, cross-sell nudges."
        else:
            idea = "Personalized picks, onboarding journey, nudge 2nd/3rd purchase."

        out.append({
            "Cluster": int(c),
            "Name": name,
            "Avg Recency (days)": round(row.get("RecencyDays", np.nan), 1) if pd.notna(row.get("RecencyDays", np.nan)) else np.nan,
            "Avg Frequency": round(row.get("Frequency", np.nan), 2),
            "Avg Monetary": round(row.get("Monetary", np.nan), 2),
            "Playbook": idea
        })
    ins = pd.DataFrame(out).sort_values("Cluster")
    return ins, labels

# ----------------------------
# I/O
# ----------------------------
@st.cache_data(show_spinner=False)
def read_uploaded_file(_file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Leading underscore in _file_bytes prevents Streamlit from hashing the raw memoryview."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        import io
        return pd.read_excel(io.BytesIO(_file_bytes), engine="openpyxl")
    return pd.read_csv(io.BytesIO(_file_bytes))

# ----------------------------
# UI starts here
# ----------------------------
st.set_page_config(page_title="Customer Segmentation & Insights (Single-File)", layout="wide")
st.title("Customer Segmentation & Insights Dashboard (Single-File)")

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

# Read file
try:
    raw = read_uploaded_file(file.getbuffer(), file.name)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.subheader("Raw preview")
st.dataframe(raw.head(10), use_container_width=True)

# Column mapping (auto + UI)
st.subheader("Column mapping")
proposed_map, scores = infer_column_map(raw)

with st.expander("Edit/confirm mapping", expanded=True):
    mapping: Dict[str, Optional[str]] = {}
    choices = ["— none —"] + list(raw.columns)
    # Required first, then optional
    for canon in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        default = proposed_map.get(canon)
        idx = choices.index(default) if default in choices else 0
        sel = st.selectbox(f"{canon}", options=choices, index=idx, key=f"map_{canon}")
        mapping[canon] = None if sel == "— none —" else sel

# Validate required
missing_required = [c for c in REQUIRED_COLUMNS if not mapping.get(c)]
if missing_required:
    st.error(f"Please map required columns: {', '.join(missing_required)}")
    st.stop()

# Apply mapping for preview
raw_mapped = apply_column_map(raw, {k: v for k, v in mapping.items() if v})
st.caption("Preview after applying mapping")
st.dataframe(raw_mapped.head(8), use_container_width=True)

# Clean + feature engineering
@st.cache_data(show_spinner=False)
def clean_and_featurize(raw_df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = clean_transactions(raw_df, column_map=mapping)
    feats = build_customer_features(tx)
    return tx, feats

with st.spinner("Cleaning & feature engineering..."):
    tx, feats = clean_and_featurize(raw, {k: v for k, v in mapping.items() if v})

# QC snapshot + downloads
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

# Features preview + core features
st.subheader("Customer Features")
st.write(f"Shape: {feats.shape}")
st.dataframe(feats.head(10), use_container_width=True)

core_cols = get_core_feature_columns(feats)
if len(core_cols) < 3 or feats.shape[0] < max(10, 2 * k_min):
    st.warning("Dataset looks small or missing key features. Consider widening k-range or ensuring enough customers.")

# Scale + cluster
X_scaled, _ = scale_features(feats, core_cols)
with st.spinner("Selecting k via silhouette & clustering..."):
    best = kmeans_with_silhouette(X_scaled, k_min=k_min, k_max=k_max)
    if best["k"] is None:
        st.error("Could not find a valid k. Try a different k range or check data quality.")
        st.stop()
    feats["Cluster"] = best["labels"]

st.success(f"Selected k = {best['k']}  |  silhouette = {best['score']:.3f}")

# Profiles, charts, insights
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
