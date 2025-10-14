from __future__ import annotations
import pandas as pd

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
        return ["Tiered loyalty w/ experiences","Early access/drops","Referral incentives"]
    if label == "Churn Risk (Lapsed Value)":
        return ["Win-back bundles","Timed discount + free ship","Remind expiring credits"]
    if label == "Low-Spend Infrequents":
        return ["Low-ASP bundles/add-ons","Cart threshold nudges","Value education content"]
    if label == "New Customers":
        return ["Welcome series (90d)","Onboarding tutorials","Collect prefs for personalization"]
    return ["New-arrival nudges","Personalized recos","Occasional premium upsell"]

def build_insights_table(profile: pd.DataFrame):
    names = label_clusters(profile)
    profile = profile.copy()
    profile["Label"] = profile.index.map(names)
    rows = []
    for cid, row in profile.iterrows():
        recs = recommendations_for(row["Label"])
        rows.append({
            "Cluster": int(cid),
            "Label": row["Label"],
            "Customers": int(row["Count"]),
            "Monetary↑": round(row["Monetary"], 2),
            "RecencyDays": round(row["RecencyDays"], 1),
            "Frequency": round(row["Frequency"], 2),
            "Top actions": " | ".join(recs[:3]),
        })
    ins_df = pd.DataFrame(rows)
    return ins_df, names

# --- Phase 1 helpers (append to src/insights.py) ---
import numpy as np
import pandas as pd

def compute_business_kpis(feats: pd.DataFrame) -> dict:
    """Return dict: customers, revenue (sum Monetary), aov, repeat_rate."""
    out = {"customers": 0, "revenue": np.nan, "aov": np.nan, "repeat_rate": np.nan}
    if feats is None or feats.empty:
        return out
    out["customers"] = int(feats.shape[0])

    if "Monetary" in feats.columns:
        rev = float(pd.to_numeric(feats["Monetary"], errors="coerce").sum())
        out["revenue"] = rev
        if "Frequency" in feats.columns:
            orders = float(pd.to_numeric(feats["Frequency"], errors="coerce").sum())
            if orders > 0:
                out["aov"] = rev / orders

    if "Frequency" in feats.columns:
        freq = pd.to_numeric(feats["Frequency"], errors="coerce")
        out["repeat_rate"] = float((freq >= 2).mean()) if len(freq) else np.nan
    return out

def segment_share_tables(feats: pd.DataFrame):
    """
    Return (seg_counts, seg_revenue). seg_revenue is None if Monetary missing.
    """
    if feats is None or feats.empty or "Cluster" not in feats.columns:
        return pd.Series(dtype=int), None
    seg_counts = feats["Cluster"].value_counts().sort_index()
    seg_rev = None
    if "Monetary" in feats.columns:
        seg_rev = (
            feats.groupby("Cluster")["Monetary"]
            .sum()
            .sort_index()
        )
    return seg_counts, seg_rev

def ensure_transaction_amount(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Add TransactionAmount when possible from Quantity*UnitPrice.
    Returns original df if already present or not computable.
    """
    if tx is None or tx.empty:
        return tx
    if "TransactionAmount" in tx.columns:
        return tx
    if {"Quantity", "UnitPrice"}.issubset(tx.columns):
        txx = tx.copy()
        q = pd.to_numeric(txx["Quantity"], errors="coerce")
        p = pd.to_numeric(txx["UnitPrice"], errors="coerce")
        txx["TransactionAmount"] = q * p
        return txx
    return tx

def top_categories_per_segment(
    tx: pd.DataFrame,
    feats: pd.DataFrame,
    candidates: list[str] | None = None,
    top_n: int = 5,
) -> pd.DataFrame | None:
    """
    Returns a DataFrame with columns [Cluster, Category, TransactionAmount]
    containing the top-N categories per cluster. None if not computable.
    """
    if tx is None or feats is None or tx.empty or feats.empty:
        return None
    if "CustomerID" not in tx.columns or "Cluster" not in feats.columns:
        return None

    if candidates is None:
        candidates = ["Description", "StockCode", "Category", "Sub-Category", "Product", "ProductName"]

    tx = ensure_transaction_amount(tx)
    if "TransactionAmount" not in tx.columns:
        return None

    cat_col = next((c for c in candidates if c in tx.columns), None)
    if cat_col is None:
        return None

    merged = tx.merge(feats[["CustomerID", "Cluster"]], on="CustomerID", how="left")
    cat_agg = (
        merged.groupby(["Cluster", cat_col])["TransactionAmount"]
        .sum()
        .reset_index()
        .rename(columns={cat_col: "Category"})
    )

    topN = (
        cat_agg.sort_values(["Cluster", "TransactionAmount"], ascending=[True, False])
              .groupby("Cluster")
              .head(top_n)
              .reset_index(drop=True)
    )
    return topN
