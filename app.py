"""
Smart Grid Load Forecasting Dashboard
Run with: streamlit run app.py
Place this file inside the same folder as: nldc_combined.csv, features.csv,
model_comparison_with_lstm.csv, predictions.csv
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="Smart Grid Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------
# Data loading (cached so it only reads from disk once)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    raw = pd.read_csv(os.path.join(base, "nldc_combined.csv"), index_col=0, parse_dates=True)
    features = pd.read_csv(os.path.join(base, "features.csv"), index_col=0, parse_dates=True)
    comparison = pd.read_csv(os.path.join(base, "model_comparison_with_lstm.csv"))
    predictions = pd.read_csv(os.path.join(base, "predictions.csv"), index_col=0, parse_dates=True)
    return raw, features, comparison, predictions

raw, features, comparison, predictions = load_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚡ Smart Grid Dashboard")
st.sidebar.markdown("**NLDC All-India Demand — August 2026**")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Exploratory Analysis", "Model Comparison", "Forecast Viewer"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {raw.index.min().date()} to {raw.index.max().date()}")
st.sidebar.caption(f"{len(raw):,} records @ 15-min resolution")
st.sidebar.caption(f"Model Trained by Rowsan ")
# ---------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------
if page == "Overview":
    st.title("Smart Grid Load Forecasting — Overview")
    st.markdown("BTech EE Final Year Project | ML/DL-based demand forecasting using NLDC India data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Demand", f"{raw['DEMAND_MET_MW'].max():,.0f} MW")
    c2.metric("Min Demand", f"{raw['DEMAND_MET_MW'].min():,.0f} MW")
    c3.metric("Avg Demand", f"{raw['DEMAND_MET_MW'].mean():,.0f} MW")
    c4.metric("Days Covered", f"{raw.index.normalize().nunique()}")

    st.subheader("Full Month Demand Curve")
    fig = px.line(raw, x=raw.index, y="DEMAND_MET_MW",
                  labels={"x": "Date", "DEMAND_MET_MW": "MW"})
    fig.update_traces(line=dict(width=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Generation Mix (avg. MW by source)")
    gen_cols = ["NUCLEAR_MW", "WIND_MW", "SOLAR_MW", "HYDRO_MW", "GAS_MW", "THERMAL_MW"]
    gen_avg = raw[gen_cols].mean().sort_values(ascending=False)
    fig2 = px.bar(gen_avg, orientation="h", labels={"value": "Avg MW", "index": "Source"})
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: Exploratory Analysis
# ---------------------------------------------------------
elif page == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    df = raw.copy()
    df["hour"] = df.index.hour + df.index.minute / 60
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["day_name"] = df.index.day_name()

    st.subheader("Average Daily Load Profile")
    hourly = df.groupby("hour")["DEMAND_MET_MW"].mean().reset_index()
    fig = px.line(hourly, x="hour", y="DEMAND_MET_MW",
                  labels={"hour": "Hour of Day", "DEMAND_MET_MW": "Avg MW"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday vs Weekend Profile")
    wk = df[~df["is_weekend"]].groupby("hour")["DEMAND_MET_MW"].mean()
    we = df[df["is_weekend"]].groupby("hour")["DEMAND_MET_MW"].mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=wk.index, y=wk.values, name="Weekday"))
    fig2.add_trace(go.Scatter(x=we.index, y=we.values, name="Weekend"))
    fig2.update_layout(xaxis_title="Hour of Day", yaxis_title="Avg MW")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Demand Distribution by Day of Week")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig3 = px.box(df, x="day_name", y="DEMAND_MET_MW", category_orders={"day_name": order})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Correlation with Demand")
    num_cols = ["FREQUENCY_Hz", "DEMAND_MET_MW", "NUCLEAR_MW", "WIND_MW", "SOLAR_MW",
                "HYDRO_MW", "GAS_MW", "THERMAL_MW", "STORAGE_MW", "TOTAL_GENERATION_MW"]
    corr = raw[num_cols].corr()["DEMAND_MET_MW"].drop("DEMAND_MET_MW").sort_values()
    fig4 = px.bar(corr, orientation="h", labels={"value": "Correlation", "index": ""})
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: Model Comparison
# ---------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.dataframe(
        comparison.style.highlight_min(subset=["MAE", "RMSE", "MAPE(%)"], color="lightgreen")
                         .highlight_max(subset=["R2"], color="lightgreen"),
        use_container_width=True
    )

    metric = st.selectbox("Compare models by:", ["MAE", "RMSE", "MAPE(%)", "R2"])
    fig = px.bar(comparison.sort_values(metric), x="Model", y=metric, color="Model")
    st.plotly_chart(fig, use_container_width=True)

    best_model = comparison.sort_values("RMSE").iloc[0]
    st.success(f"Best performing model: **{best_model['Model']}** "
               f"(RMSE = {best_model['RMSE']:.1f} MW, MAPE = {best_model['MAPE(%)']:.2f}%)")

# ---------------------------------------------------------
# PAGE 4: Forecast Viewer
# ---------------------------------------------------------
elif page == "Forecast Viewer":
    st.title("Actual vs Predicted Demand")

    model_cols = [c for c in predictions.columns if c != "Actual"]
    chosen = st.multiselect("Select model(s) to display", model_cols, default=model_cols[:2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions["Actual"],
                              name="Actual", line=dict(color="black", width=2)))
    for m in chosen:
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[m], name=m,
                                  line=dict(width=1)))
    fig.update_layout(xaxis_title="Date", yaxis_title="MW", height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Error Distribution")
    if chosen:
        errs = pd.DataFrame({
            m: predictions["Actual"] - predictions[m] for m in chosen
        })
        fig2 = px.histogram(errs, barmode="overlay", opacity=0.6,
                             labels={"value": "Error (MW)"})
        st.plotly_chart(fig2, use_container_width=True)
