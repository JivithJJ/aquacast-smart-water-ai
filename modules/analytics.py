import pandas as pd
from datetime import timedelta

def compute_split(df):
    total = df[["borewell_litres","bwssb_litres","tanker_litres"]].sum().sum()
    return {
        "Borewell %": round(df["borewell_litres"].sum()/total*100, 2),
        "BWSSB %": round(df["bwssb_litres"].sum()/total*100, 2),
        "Tanker %": round(df["tanker_litres"].sum()/total*100, 2),
        "Total Litres": int(total)
    }

def get_source_split(df):
    last = df["date"].max()
    w7 = df[df["date"] > last - timedelta(days=7)]
    w30 = df[df["date"] > last - timedelta(days=30)]
    w90 = df[df["date"] > last - timedelta(days=90)]

    table = pd.DataFrame([
        {"Window": "7 Days", **compute_split(w7)},
        {"Window": "30 Days", **compute_split(w30)},
        {"Window": "90 Days", **compute_split(w90)}
    ])

    return table