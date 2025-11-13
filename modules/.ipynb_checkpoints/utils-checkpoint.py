import pandas as pd

def load_dataset(path="../data/aquacast_whitefield_3year.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def add_lag_features(df, target_col="tanker_litres", lags=[1,2,3,7,14,21]):
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    df = df.fillna(method="bfill").fillna(0)
    return df

def calc_tankers_required(litres, tanker_size=12000):
    import math
    return int(math.ceil(litres / tanker_size))