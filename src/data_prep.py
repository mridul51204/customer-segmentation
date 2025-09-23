from __future__ import annotations
import re, difflib
import pandas as pd
import numpy as np

# Canonical schema we use everywhere
REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
OPTIONAL_COLUMNS = ["CustomerID", "Country", "StockCode", "Description"]
CANONICAL = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Name normalization
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.strip().lower())

# Synonyms by normalized form
SYN = {
    "InvoiceNo":   ["invoiceno","invoice","invoiceid","invoicenumber","orderno","orderid","billno","billingid"],
    "InvoiceDate": ["invoicedate","date","orderdate","datetime","timestamp","docdate"],
    "Quantity":    ["quantity","qty","itemqty","itemquantity","units","count","noofitems"],
    "UnitPrice":   ["unitprice","price","unit_price","unitcost","unit_cost","rate","amountperunit","itemprice"],
    "CustomerID":  ["customerid","customer_id","customer id","userid","user_id","user id","clientid","memberid"],
    "Country":     ["country","countryname","nation","region"],
    "StockCode":   ["stockcode","sku","productcode","product_code","itemcode","itemid"],
    "Description": ["description","productname","itemname","product_name","item_name","title","desc"],
}

def infer_column_map(df: pd.DataFrame, min_ratio: float = 0.6) -> tuple[dict, dict]:
    """
    Returns (mapping, scores) where mapping is {Canonical -> ActualColumnName}.
    Tries exact/synonym matches first; then fuzzy.
    """
    cols = list(df.columns)
    norm2orig = {_norm(c): c for c in cols}

    mapping: dict[str, str] = {}
    scores: dict[str, float] = {}

    # 1) direct synonym hits
    for canon in CANONICAL:
        for syn in SYN[canon]:
            if syn in norm2orig:
                mapping[canon] = norm2orig[syn]
                scores[canon] = 1.0
                break

    # 2) fuzzy name matching for unmapped
    for canon in CANONICAL:
        if canon in mapping:
            continue
        candidates = difflib.get_close_matches(canon.lower(), [_norm(c) for c in cols], n=3, cutoff=min_ratio)
        best = None
        best_ratio = 0.0
        for cand_norm in candidates:
            orig = norm2orig.get(cand_norm)
            if not orig:
                continue
            r = difflib.SequenceMatcher(None, canon.lower(), cand_norm).ratio()
            if r > best_ratio:
                best, best_ratio = orig, r
        if best and best_ratio >= min_ratio:
            mapping[canon] = best
            scores[canon] = best_ratio

    return mapping, scores

def apply_column_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename df columns to canonical names using the provided mapping {Canonical->Actual}."""
    inv_map = {v: k for k, v in mapping.items() if v in df.columns}
    return df.rename(columns=inv_map)

def clean_transactions(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Standard clean + compute LineRevenue. Auto-maps columns if column_map is None."""
    df = df.copy()

    # Auto-map
    if column_map is None:
        column_map, _ = infer_column_map(df)
    df = apply_column_map(df, column_map)

    # Parse dtypes
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

    # Drop nulls in criticals (only if present)
    need = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if need:
        df = df.dropna(subset=need)

    # Remove cancellations/negatives
    if "InvoiceNo" in df.columns and df["InvoiceNo"].dtype == "object":
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    # Revenue
    if {"Quantity", "UnitPrice"}.issubset(df.columns):
        df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    keep = [c for c in ["InvoiceNo","StockCode","Description","Quantity",
                        "InvoiceDate","UnitPrice","CustomerID","Country",
                        "LineRevenue"] if c in df.columns]
    df = df[keep].drop_duplicates().sort_values("InvoiceDate")
    return df.reset_index(drop=True)
