import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import timedelta
from modules.utils import add_lag_features

# ------------------------------
# Load models & metadata
# ------------------------------
with open(".../models/sarimax_model.pkl", "rb") as f:
    sar_model = pickle.load(f)

gbm_model = joblib.load("../models/gbm_model.pkl")

with open("../models/model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_cols = metadata["feature_cols"]
res_cols = metadata["residual_feature_cols"]

# ------------------------------
# 7-day rolling forecast
# ------------------------------
def forecast_next_7_days(df):
    df_fc = df.copy()
    preds = []
    dates = []

    last_date = df_fc["date"].max()

    for step in range(7):
        df_fc = add_lag_features(df_fc)

        latest = df_fc.iloc[-1]

        X_sar = latest[feature_cols].values.reshape(1, -1)
        X_gbm = latest[res_cols].values.reshape(1, -1)

        sar_pred = sar_model.predict(start=sar_model.nobs, end=sar_model.nobs, exog=X_sar).iloc[0]
        gbm_pred = gbm_model.predict(X_gbm)[0]

        final_pred = sar_pred + gbm_pred
        preds.append(final_pred)

        next_date = last_date + timedelta(days=step+1)
        dates.append(next_date)

        new_row = latest.copy()
        new_row["date"] = next_date
        new_row["tanker_litres"] = final_pred

        df_fc = pd.concat([df_fc, new_row.to_frame().T], ignore_index=True)

    return dates, preds

# ------------------------------
# 30-day forecast (budget)
# ------------------------------
def forecast_next_30_days(df):
    df_fc = df.copy()
    preds = []
    dates = []

    last_date = df_fc["date"].max()

    for step in range(30):
        df_fc = add_lag_features(df_fc)

        latest = df_fc.iloc[-1]

        X_sar = latest[feature_cols].values.reshape(1, -1)
        X_gbm = latest[res_cols].values.reshape(1, -1)

        sar_pred = sar_model.predict(start=sar_model.nobs, end=sar_model.nobs, exog=X_sar).iloc[0]
        gbm_pred = gbm_model.predict(X_gbm)[0]

        final_pred = sar_pred + gbm_pred
        preds.append(final_pred)

        next_date = last_date + timedelta(days=step+1)
        dates.append(next_date)

        new_row = latest.copy()
        new_row["date"] = next_date
        new_row["tanker_litres"] = final_pred

        df_fc = pd.concat([df_fc, new_row.to_frame().T], ignore_index=True)

    return dates, preds