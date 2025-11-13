# app.py - AquaCast Streamlit Dashboard
# Place this file at the project root (AquaCast/app.py)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import io

# modules (from your modules/ folder)
from modules.utils import load_dataset, calc_tankers_required
from modules import forecast as fc
from modules import pricing as pr
from modules import analytics as an

st.set_page_config(page_title="AquaCast • Smart Water Forecasting", page_icon="💧", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = load_dataset("data/aquacast_whitefield_3year.csv")
    return df

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Load dataset (cached)
# -------------------------
df = load_data()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("AquaCast")
st.sidebar.caption("Smart water forecasting & cost optimization")
page = st.sidebar.radio("Navigate", ["Home", "7-Day Forecast", "30-Day Budget Planner", "Source Split", "Export & Notes"])




# -------------------------
# Home
# -------------------------
if page == "Home":
    st.title(" AquaCast — Smart Water Intelligence")
    st.markdown(
        """
        AquaCast combines time-series forecasting (SARIMAX) + ML residual correction (GBM),
        plus a dynamic pricing model to produce operational insights for apartment RWAs:
        - 7-day tanker demand forecast (tankers = ceil(volume/12kL)
        - 30-day budget planner with pre-book savings
        - Source split analysis (Borewell / BWSSB / Tanker)
        - CSV export and downloadable reports
        """
    )

    # Key metrics
    last_date = df["date"].max().date()
    recent_mean = int(df.tail(7)["tanker_litres"].mean())
    st.metric("Data last date", f"{last_date}")
    st.metric("Recent 7-day avg tanker litres", f"{recent_mean:,} L")

    st.markdown("### Quick visual: tanker litres (historical)")
    chart_df = df[["date", "tanker_litres"]].set_index("date").resample("D").sum()
    st.line_chart(chart_df["tanker_litres"])

# -------------------------
# 7-Day Forecast page
# -------------------------
if page == "7-Day Forecast":
    st.title(" 7-Day Tanker Forecast")
    st.markdown("Model: Hybrid SARIMAX + GBM residual correction. Output shows predicted litres and tankers required.")

    with st.spinner("Running 7-day forecast..."):
        try:
            dates7, preds7 = fc.forecast_next_7_days(df)
        except Exception as e:
            st.error("Forecast failed. See error below and check models/ and modules/.")
            st.exception(e)
            st.stop()

    df7 = pd.DataFrame({"date": pd.to_datetime(dates7), "forecast_litres": np.round(preds7, 1)})
    df7["tankers_needed"] = df7["forecast_litres"].apply(lambda x: calc_tankers_required(x, 12000))
    df7["weekday"] = df7["date"].dt.day_name()

    st.dataframe(df7, use_container_width=True)

    # Simple plot
    st.markdown("### Forecast litres (next 7 days)")
    fig = pd.DataFrame({"date": df7["date"].dt.date, "litres": df7["forecast_litres"]}).set_index("date")
    st.bar_chart(fig["litres"])

    # Download button
    csv_bytes = df_to_csv_bytes(df7)
    st.download_button("Download 7-day forecast CSV", csv_bytes, file_name="aquacast_7day_forecast.csv", mime="text/csv")

# -------------------------
# 30-Day Budget Planner
# -------------------------
if page == "30-Day Budget Planner":
    st.title(" 30-Day Budget Planner")
    st.markdown("Generates day-wise market vs pre-book cost and savings for next 30 days.")

    with st.spinner("Running 30-day forecast + cost table..."):
        try:
            dates30, preds30 = fc.forecast_next_30_days(df)
            cost_df = pr.build_cost_table(dates30, preds30, df)
        except Exception as e:
            st.error("30-day budget generation failed. Check saved models and module code.")
            st.exception(e)
            st.stop()

    # show table
    st.dataframe(cost_df, use_container_width=True)

    # summary metrics
    market_total = cost_df["market_cost_rs"].sum()
    ref_total = cost_df["prebook_cost_rs"].sum()
    savings = round(market_total - ref_total, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total market cost (30d)", f"₹{market_total:,.2f}")
    c2.metric("Total prebook cost (30d)", f"₹{ref_total:,.2f}")
    c3.metric("Estimated savings (30d)", f"₹{savings:,.2f}")

    # highlight top 5 saving days
    st.markdown("### Top saving days (by savings_rs)")
    st.table(cost_df.sort_values("savings_rs", ascending=False).head(5).set_index("date"))

    # charts
    st.markdown("### Market vs Prebook cost (next 30 days)")
    chart_cost = cost_df.set_index("date")[["market_cost_rs", "prebook_cost_rs"]]
    st.line_chart(chart_cost)

    # CSV download
    st.download_button("Download 30-day budget CSV", df_to_csv_bytes(cost_df), file_name="aquacast_30day_budget.csv", mime="text/csv")

# -------------------------
# Source Split Analysis
# -------------------------
if page == "Source Split":
    st.title(" Source Split (7 / 30 / 90 days)")
    st.markdown("Shows share of water coming from Borewell, BWSSB and Tankers.")

    table = an.get_source_split(df)
    st.dataframe(table, use_container_width=True)

    # pie charts for 30 day
    st.markdown("### 30-day split visualization")
    try:
        row30 = table[table["Window"] == "30 Days"].iloc[0]
        labels = ["Borewell", "BWSSB", "Tanker"]
        values = [row30["Borewell %"], row30["BWSSB %"], row30["Tanker %"]]
        fig = pd.DataFrame({"source": labels, "pct": values}).set_index("source")
        st.bar_chart(fig["pct"])
    except Exception:
        st.info("Not enough data to plot split chart.")

    # notes to RWA
    st.markdown("### Actionable notes (auto-generated)")
    if row30["Tanker %"] > 30:
        st.warning("High tanker dependency (>30%). Consider borewell maintenance and rainwater harvesting incentives.")
    else:
        st.success("Tanker dependency is moderate.")

# -------------------------
# Export & Notes
# -------------------------
if page == "Export & Notes":
    st.title("📦 Export & Developer Notes")
    st.markdown("Download the core CSVs and models, plus quick instructions.")

    st.markdown("**Datasets available in `data/`**")
    try:
        st.write(df[["date","tanker_litres","bwssb_litres","borewell_litres"]].tail(10))
    except Exception:
        st.write("Dataset preview unavailable.")

    # Export existing CSVs
    try:
        csv30 = pd.read_csv("data/aquacast_30day_budget_projection.csv")
        st.download_button("Download saved 30-day CSV (project)", csv30.to_csv(index=False).encode("utf-8"), file_name="aquacast_30day_budget_projection.csv")
    except Exception:
        st.info("No saved 30-day CSV found in data/")

    st.markdown("---")
    st.markdown("**Run locally**\n\n```bash\nstreamlit run app.py\n```")

    st.markdown("**Developer notes**")
    st.markdown(
        """
        - Modules live in /modules: forecast.py, pricing.py, analytics.py, utils.py  
        - Models are expected in /models: sarimax_model.pkl, gbm_model.pkl, model_metadata.pkl  
        - If you update the models, save them with the same names (see notebook).  
        - For production deployment, consider precomputing forecasts daily and serving cached files.
        """
    )

# -------------------------
# Footer
# -------------------------
st.sidebar.markdown("---")
st.sidebar.caption("AquaCast • Demo (For interviews & portfolios)")