from __future__ import annotations
import pandas as pd

def label_clusters(profile: pd.DataFrame) -> dict:
    """
    Assign human-readable labels to clusters using quantile thresholds.
    Safe if some columns (e.g., TenureDays) are missing or K is small.
    Expects profile indexed by cluster with columns: RecencyDays, Frequency, Monetary, optional TenureDays.
    """
    prof = profile.copy()

    # keep only numeric columns we use and that exist
    cols = [c for c in ["RecencyDays", "Frequency", "Monetary", "TenureDays"] if c in prof.columns]
    if not {"RecencyDays", "Frequency", "Monetary"}.issubset(cols):
        # minimal set not present → default to "Regulars"
        return {cid: "Mid-Tier Regulars" for cid in prof.index}

    # quantiles with NaN-safe fallback
    q = prof[cols].quantile([0.33, 0.66], numeric_only=True).fillna(method="ffill").fillna(method="bfill")

    names = {}
    for cid, row in prof.iterrows():
        R, F, M = row.get("RecencyDays", float("nan")), row.get("Frequency", float("nan")), row.get("Monetary", float("nan"))
        T = row.get("TenureDays", float("nan"))

        # booleans with NaN-safe comparisons
        hi_M = (pd.notna(M) and M >= q.loc[0.66, "Monetary"])
        hi_F = (pd.notna(F) and F >= q.loc[0.66, "Frequency"])
        low_R = (pd.notna(R) and R <= q.loc[0.33, "RecencyDays"])
        hi_R  = (pd.notna(R) and R >= q.loc[0.66, "RecencyDays"])
        low_T = ("TenureDays" in q.columns) and pd.notna(T) and T <= q.loc[0.33, "TenureDays"]

        if hi_M and hi_F and low_R:
            label = "High-Value Loyalists"
        elif hi_R and (pd.notna(M) and M >= q.loc[0.33, "Monetary"]):
            label = "Churn Risk (Lapsed Value)"
        elif (pd.notna(F) and F <= q.loc[0.33, "Frequency"]) and (pd.notna(M) and M <= q.loc[0.33, "Monetary"]):
            label = "Low-Spend Infrequents"
        elif low_T:
            label = "New Customers"
        else:
            label = "Mid-Tier Regulars"

        names[cid] = label
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
    prof = profile.copy()
    prof["Label"] = prof.index.map(names)

    if "Count" not in prof.columns:
        # best-effort fallback: use a 0/unknown count
        prof["Count"] = 0

    rows = []
    for cid, row in prof.iterrows():
        recs = recommendations_for(row["Label"])
        rows.append({
            "Cluster": int(cid),
            "Label": row["Label"],
            "Customers": int(row["Count"]) if pd.notna(row["Count"]) else 0,
            "Monetary↑": round(float(row.get("Monetary", float("nan"))), 2),
            "RecencyDays": round(float(row.get("RecencyDays", float("nan"))), 1),
            "Frequency": round(float(row.get("Frequency", float("nan"))), 2),
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
