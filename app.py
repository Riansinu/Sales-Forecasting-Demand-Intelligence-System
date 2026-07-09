import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("Sales Forecasting & Demand Intelligence System")

# -------------------------
# Check output files
# -------------------------

required_files = [
    "outputs/model_metrics.csv",
    "outputs/forecast_results.csv",
    "outputs/cluster_results.csv",
    "outputs/anomaly_results.csv"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error("Please run analysis.ipynb first.")
    st.write("Missing files:")
    for f in missing:
        st.write("-", f)
    st.stop()

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    dayfirst=True
)

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month_name()

# -------------------------
# Load Notebook Outputs
# -------------------------

forecast = pd.read_csv("outputs/forecast_results.csv")

metrics = pd.read_csv("outputs/model_metrics.csv")

clusters = pd.read_csv("outputs/cluster_results.csv")

anomalies = pd.read_csv("outputs/anomaly_results.csv")

# -------------------------
# Sidebar
# -------------------------

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# =====================================================
# SALES OVERVIEW
# =====================================================

if page == "Sales Overview":

    st.header("Sales Overview")

    yearly = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,4))

    sns.barplot(
        data=yearly,
        x="Year",
        y="Sales",
        ax=ax
    )

    st.pyplot(fig)

    monthly = (
        df.set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        monthly["Order Date"],
        monthly["Sales"],
        linewidth=2
    )

    ax.set_title("Monthly Sales")

    st.pyplot(fig)

    st.subheader("Filter Data")

    category = st.selectbox(
        "Category",
        sorted(df["Category"].unique())
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].unique())
    )

    filtered = df[
        (df["Category"] == category) &
        (df["Region"] == region)
    ]

    st.dataframe(filtered)

# =====================================================
# FORECAST
# =====================================================

elif page == "Forecast Explorer":

    st.header("Forecast Explorer")

    horizon = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    best_model = metrics.loc[
        metrics["RMSE"].idxmin()
    ]

    st.metric(
        "Best Model",
        best_model["Model"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        round(best_model["MAE"],2)
    )

    c2.metric(
        "RMSE",
        round(best_model["RMSE"],2)
    )

    c3.metric(
        "MAPE",
        round(best_model["MAPE"],2)
    )

    st.subheader("Forecast Results")

    st.dataframe(forecast)

# =====================================================
# ANOMALIES
# =====================================================

elif page == "Anomaly Report":

    st.header("Anomaly Detection")

    weekly = (
        df.set_index("Order Date")
        .resample("W")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        weekly["Order Date"],
        weekly["Sales"]
    )

    ax.set_title("Weekly Sales")

    st.pyplot(fig)

    st.subheader("Detected Anomalies")

    st.dataframe(anomalies)

# =====================================================
# CLUSTERS
# =====================================================

elif page == "Demand Segments":

    st.header("Product Demand Segments")

    st.dataframe(clusters)