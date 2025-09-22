# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation & Insights", layout="wide")

# ----------------------------
# Helpers: cleaning + features
# ----------------------------
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "CustomerID" in df.columns:
        try:
            df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")
        except Exception:
            df["CustomerID"] = df["CustomerID"].astype("string")

    need = [c for c in ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)

    if "InvoiceNo" in df.columns and df["InvoiceNo"].dtype == "object":
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    if {"Quantity", "UnitPrice"}.issubset(df.columns):
        df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    keep = [c for c in ["InvoiceNo","StockCode","Description","Quantity",
                        "InvoiceDate","UnitPrice","CustomerID","Country",
                        "LineRevenue"] if c in df.columns]
    df = df[keep].drop_duplicates().sort_values("InvoiceDate")
    return df.reset_index(drop=True)

def build_customer_features(trans: pd.DataFrame) -> pd.DataFrame:
    t = trans.copy()
    if "CustomerID" not in t.columns:
        t["CustomerID"] = t["InvoiceNo"]
    t = t.dropna(subset=["InvoiceNo","InvoiceDate"])
    snap = (pd.to_datetime(t["InvoiceDate"].max()) + pd.Timedelta(days=1)).normalize()

    grp = t.groupby("CustomerID", dropna=True)
    first_purchase = grp["InvoiceDate"].min()
    last_purchase  = grp["InvoiceDate"].max()
    invoices = grp["InvoiceNo"].nunique()
    qty_sum  = grp["Quantity"].sum() if "Quantity" in t.columns else pd.Series(0, index=invoices.index)
    rev_sum  = grp["LineRevenue"].sum() if "LineRevenue" in t.columns else pd.Series(0, index=invoices.index)
    items_per_invoice = qty_sum / invoices.replace(0, np.nan)

    recency_days = (snap - last_purchase).dt.days
    tenure_days  = (snap - first_purchase).dt.days
    unique_products = grp["StockCode"].nunique() if "StockCode" in t.columns else invoices

    df = pd.DataFrame({
        "CustomerID": first_purchase.index.astype("string"),
        "RecencyDays": recency_days.values,
        "Frequency": invoices.values,
        "Monetary": rev_sum.values,
        "TenureDays": tenure_days.values,
        "AvgBasketQty": items_per_invoice.fillna(0).values,
        "UniqueProducts": unique_products.values,
    })

    if "Country" in t.columns:
        modes = (t.dropna(subset=["CustomerID"])
                   .groupby("CustomerID")["Country"]
                   .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[-1]))
        df = df.merge(modes.rename("Country"), left_on="CustomerID", right_index=True, how="left")
        dums = pd.get_dummies(df["Country"], prefix="Country")
        df = pd.concat([df.drop(columns=["Country"]), dums], axis=1)

    return df.replace([pd.NA, np.inf, -np.inf], 0).fillna(0)

def kmeans_with_silhouette(X: np.ndarray, k_min: int = 3, k_max: int = 6, random_state: int = 42):
    best = {"k": None, "score": -1, "labels": None, "model": None}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2 or min(np.bincount(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best["score"]:
            best = {"k": k, "score": score, "labels": labels, "model": km}
    return best

def label_clusters(profile: pd.DataFrame) -> dict:
    q = profile[["RecencyDays","Frequency","Monetary","TenureDays"]].quantile([0.33,0.66])
    names = {}
    for cid, row in profile.iterrows():
        if (row["Monetary"] >= q.loc[0.66,"Monetary"]) and (row["Frequency"] >= q.loc[0.66,"Frequency"]) and (row["RecencyDays"] <= q.loc[0.33,"RecencyDays"]):
            names[cid] = "High-Value Loyalists"
        elif (row["RecencyDays"] >= q.loc[0.66,"RecencyDays"]) and (row["Monetary"] >= q.loc[0.33,"Monetary"]):
            names[cid] = "Churn Risk (Lapsed Value)"
        elif (row["Frequency"] <= q.loc[0.33,"Frequency"]) and (row["Monetary"] <= q.loc[0.33,"Monetary"]):
            names[cid] = "Low-Spend Infrequents"
        elif (row["TenureDays"] <= q.loc[0.33,"TenureDays"]):
            names[cid] = "New Customers"
        else:
            names[cid] = "Mid-Tier Regulars"
    return names

def recommendations_for(label: str) -> list[str]:
    if label == "High-Value Loyalists":
        return [
            "Tiered loyalty with experiential rewards",
            "Early access to new products",
            "Referral incentives for lookalikes"
        ]
    if label == "Churn Risk (Lapsed Value)":
        return [
            "Win-back campaign with bundles",
            "Time-bound discount + free shipping",
            "Reminders about expiring credits"
        ]
    if label == "Low-Spend Infrequents":
        return [
            "Low-ASP bundles and add-ons",
            "Cart threshold nudges (₹X for free shipping)",
            "Content to educate value/usage"
        ]
    if label == "New Customers":
        return [
            "Welcome series + first-90-day nurture",
            "Onboarding tutorials/checklist",
            "Collect preferences for personalization"
        ]
    return [
        "Periodic new-arrival nudges",
        "Personalized category recommendations",
        "Occasional upsell to premium SKUs"
    ]

# ----------------------------
# UI
# ----------------------------
st.title("Customer Segmentation & Insights Dashboard")
st.write("Upload a transactions CSV/XLSX (e.g., Online Retail). The app will clean, build RFM-style features, auto-select k (3–6) by silhouette, and generate consulting-style insights.")

with st.sidebar:
    st.header("1) Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    st.header("2) Clustering options")
    k_min = st.number_input("k min", 2, 10, 3, 1)
    k_max = st.number_input("k max", 2, 15, 6, 1)
    if k_max < k_min:
        st.error("k max must be ≥ k min")

if file is None:
    st.info("👆 Upload a file to begin. Expected columns: InvoiceNo, Quantity, UnitPrice, InvoiceDate (CustomerID/Country optional).")
    st.stop()

try:
    if file.name.lower().endswith((".xlsx",".xls")):
        raw = pd.read_excel(file)
    else:
        raw = pd.read_csv(file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.subheader("Raw preview")
st.dataframe(raw.head(10), use_container_width=True)

with st.spinner("Cleaning & feature engineering..."):
    tx = clean_transactions(raw)
    feats = build_customer_features(tx)

if feats.shape[0] < max(10, k_max*3):
    st.warning("Dataset has few customers after cleaning. Results may be noisy.")

st.subheader("Customer Features (per CustomerID)")
st.write(f"Shape: {feats.shape}")
st.dataframe(feats.head(10), use_container_width=True)

core_cols = [c for c in ["RecencyDays","Frequency","Monetary","TenureDays","AvgBasketQty","UniqueProducts"] if c in feats.columns]
X = feats[core_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with st.spinner("Selecting k via silhouette & clustering..."):
    best = kmeans_with_silhouette(X_scaled, k_min=k_min, k_max=k_max)
    if best["k"] is None:
        st.error("Could not find a valid k. Try a different k range or check data quality.")
        st.stop()
    labels = best["labels"]
    feats["Cluster"] = labels

st.success(f"Selected k = {best['k']}  |  silhouette = {best['score']:.3f}")

profile = (feats.groupby("Cluster")[core_cols]
           .mean()
           .sort_index())
counts = feats["Cluster"].value_counts().sort_index()
profile["Count"] = counts.values
named = label_clusters(profile)
profile["Label"] = profile.index.map(named)

colA, colB = st.columns([1,1])
with colA:
    st.subheader("Cluster Sizes")
    fig_ct = px.bar(counts, labels={"value":"Customers","index":"Cluster"})
    st.plotly_chart(fig_ct, use_container_width=True)
with colB:
    st.subheader("Avg Monetary by Cluster")
    fig_rev = px.bar(profile, x=profile.index, y="Monetary", labels={"x":"Cluster","Monetary":"Avg Monetary"})
    st.plotly_chart(fig_rev, use_container_width=True)

pca = PCA(n_components=2, random_state=42)
pts2 = pca.fit_transform(X_scaled)
vis = pd.DataFrame(pts2, columns=["PC1","PC2"])
vis["Cluster"] = labels
fig_sc = px.scatter(vis, x="PC1", y="PC2", color="Cluster",
                    title="Clusters (PCA 2D Projection)", opacity=0.8)
st.plotly_chart(fig_sc, use_container_width=True)

st.subheader("RFM-style Distributions by Cluster")
for col in ["RecencyDays","Frequency","Monetary"]:
    if col in feats.columns:
        fig = px.box(feats, x="Cluster", y=col, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Insights & Recommendations")
ins_rows = []
for cid, row in profile.iterrows():
    label = row["Label"]
    recs = recommendations_for(label)
    ins_rows.append({
        "Cluster": int(cid),
        "Label": label,
        "Customers": int(row["Count"]),
        "Monetary↑": round(row["Monetary"], 2),
        "RecencyDays": round(row["RecencyDays"], 1),
        "Frequency": round(row["Frequency"], 2),
        "Top actions": " | ".join(recs[:3])
    })
ins_df = pd.DataFrame(ins_rows)
st.dataframe(ins_df, use_container_width=True)

assign = feats[["CustomerID","Cluster"]].copy()
assign["Label"] = assign["Cluster"].map(named)
st.download_button(
    "⬇️ Download customer→cluster CSV",
    assign.to_csv(index=False).encode("utf-8"),
    file_name="customer_clusters.csv",
    mime="text/csv"
)

st.caption("Tip: Use this CSV for CRM targeting, win-back campaigns, and lookalike audiences.")
