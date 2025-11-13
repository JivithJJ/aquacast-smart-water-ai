import numpy as np
import pandas as pd

BASE_PRICE = 0.14
WEEKEND_MULT = 1.06
SHORTAGE_MULT = 1.10

def compute_price_per_litre(date, bwssb_supply_index):
    price = BASE_PRICE

    if date.weekday() >= 5:
        price *= WEEKEND_MULT

    if bwssb_supply_index < 1.0:
        price *= SHORTAGE_MULT

    return price

def build_cost_table(dates, litres_forecast, df):
    ref_price = 0.14
    rows = []

    bwssb_idx = float(df.tail(7)["bwssb_supply_index"].median())

    for d, l in zip(dates, litres_forecast):
        tankers = int(np.ceil(l / 12000))

        market_price = compute_price_per_litre(d, bwssb_idx)
        cost_market = l * market_price
        cost_ref = l * ref_price

        rows.append({
            "date": d,
            "forecast_litres": round(l, 2),
            "tankers_needed": tankers,
            "market_price_per_l": round(market_price, 4),
            "market_cost_rs": round(cost_market, 2),
            "prebook_cost_rs": round(cost_ref, 2),
            "savings_rs": round(cost_market - cost_ref, 2),
            "weekend": 1 if d.weekday() >= 5 else 0
        })

    return pd.DataFrame(rows)