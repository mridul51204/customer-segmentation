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
