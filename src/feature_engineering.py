from __future__ import annotations
import pandas as pd
import numpy as np

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
