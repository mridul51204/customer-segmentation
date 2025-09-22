from __future__ import annotations
import pandas as pd
import numpy as np

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

    need = [c for c in ["InvoiceNo","InvoiceDate","Quantity","UnitPrice"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)

    if "InvoiceNo" in df.columns and df["InvoiceNo"].dtype == "object":
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    if "Quantity" in df.columns:
        df = df[df["Quantity"] > 0]
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"] > 0]

    if {"Quantity","UnitPrice"}.issubset(df.columns):
        df["LineRevenue"] = df["Quantity"] * df["UnitPrice"]

    keep = [c for c in ["InvoiceNo","StockCode","Description","Quantity",
                        "InvoiceDate","UnitPrice","CustomerID","Country",
                        "LineRevenue"] if c in df.columns]
    df = df[keep].drop_duplicates().sort_values("InvoiceDate")
    return df.reset_index(drop=True)
