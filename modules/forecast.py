# modules/forecast.py

import numpy as np
import pandas as pd
import joblib
import pickle

# ----------------------------------------
# Load models + metadata
# ----------------------------------------
with open("models/sarimax_model.pkl", "rb") as f:
    sar_model = pickle.load(f)

gbm_model = joblib.load("models/gbm_model.pkl")

with open("models/model_metadata.pkl", "rb") as f:
    meta = pickle.load(f)

feature_cols = meta["feature_cols"]
residual_feature_cols = meta["residual_feature_cols"]
train_columns = meta["train_columns"]


# ----------------------------------------
# Helper: build engineered features (keeps date safe)
# ----------------------------------------
def prepare_features(df):
    df = df.copy()

    # Ensure date is datetime (important)
    df["date"] = pd.to_datetime(df["date"])

    # weekend flag (will be numeric)
    df["is_weekend"] = df["date"].dt.weekday.isin([5, 6]).astype(int)

    # lag features based on tanker_litres
    for lag in [1, 2, 3, 7, 14, 21]:
        df[f"tanker_lag_{lag}"] = df["tanker_litres"].shift(lag)

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- Preserve date, coerce other columns to numeric safely ---
    date_col = df["date"].copy()
    other = df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = pd.concat([date_col, other], axis=1)

    return df


# ----------------------------------------
# ensure a row (Series) is numeric but keep date
# ----------------------------------------
def force_numeric_keep_date(row):
    row = row.copy()
    date_val = row.get("date", None)
    # Keep date as-is
    # Coerce other entries
    for k in row.index:
        if k == "date":
            continue
        v = row[k]
        # replace inf / -inf / nan with 0
        if pd.isna(v) or v in [np.inf, -np.inf]:
            row[k] = 0.0
            continue
        # convert to float if possible
        try:
            row[k] = float(v)
        except Exception:
            row[k] = 0.0
    # restore date
    if date_val is not None:
        row["date"] = pd.to_datetime(date_val)
    return row


# ========================================
#     7-DAY FORECAST
# ========================================
def forecast_next_7_days(df_raw):

    # 1) build features and keep date
    df = prepare_features(df_raw)

    # 2) ensure non-date columns numeric
    date_col = df["date"].copy()
    df_other = df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = pd.concat([date_col, df_other], axis=1)

    # 3) history window
    history = df.copy().tail(30).reset_index(drop=True)

    preds = []
    dates = []

    for step in range(7):
        latest = history.iloc[-1].copy()
        latest = force_numeric_keep_date(latest)

        # SARIMAX exog (force float)
        X_sar = latest[feature_cols].astype(float).values.reshape(1, -1)
        sar_pred_raw = sar_model.forecast(steps=1, exog=X_sar)
        sar_pred = float(np.array(sar_pred_raw).flatten()[0])

        # GBM residual
        X_gbm = latest[residual_feature_cols].astype(float).values.reshape(1, -1)
        gbm_res = float(gbm_model.predict(X_gbm)[0])

        final_pred = sar_pred + gbm_res
        preds.append(final_pred)

        # compute next date (latest["date"] is datetime)
        next_date = pd.to_datetime(latest["date"]) + pd.Timedelta(days=1)
        dates.append(next_date)

        # append new row for rolling history
        new_row = latest.copy()
        new_row["date"] = next_date
        new_row["tanker_litres"] = final_pred

        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)

        # rebuild engineered features on history and keep date safe
        history = prepare_features(history)
        date_col = history["date"].copy()
        history_other = history.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce").fillna(0)
        history = pd.concat([date_col, history_other], axis=1).reset_index(drop=True)

    return dates, preds


# ========================================
#     30-DAY FORECAST
# ========================================
def forecast_next_30_days(df_raw):

    df = prepare_features(df_raw)
    date_col = df["date"].copy()
    df_other = df.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    df = pd.concat([date_col, df_other], axis=1)

    history = df.copy().tail(30).reset_index(drop=True)

    preds = []
    dates = []

    for step in range(30):

        latest = history.iloc[-1].copy()
        latest = force_numeric_keep_date(latest)

        X_sar = latest[feature_cols].astype(float).values.reshape(1, -1)
        sar_pred_raw = sar_model.forecast(steps=1, exog=X_sar)
        sar_pred = float(np.array(sar_pred_raw).flatten()[0])

        X_gbm = latest[residual_feature_cols].astype(float).values.reshape(1, -1)
        gbm_res = float(gbm_model.predict(X_gbm)[0])

        final_pred = sar_pred + gbm_res
        preds.append(final_pred)

        next_date = pd.to_datetime(latest["date"]) + pd.Timedelta(days=1)
        dates.append(next_date)

        new_row = latest.copy()
        new_row["date"] = next_date
        new_row["tanker_litres"] = final_pred

        history = pd.concat([history, new_row.to_frame().T], ignore_index=True)

        history = prepare_features(history)
        date_col = history["date"].copy()
        history_other = history.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce").fillna(0)
        history = pd.concat([date_col, history_other], axis=1).reset_index(drop=True)

    return dates, preds