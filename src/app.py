"""
app.py — NeuroInsights: upload any CSV, get instant automated profiling,
visualizations, and plain-English insights.
Run: streamlit run src/app.py
"""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from profiler import profile_dataframe
from insight_generator import generate_insights
from auto_visualizer import generate_column_chart, generate_correlation_heatmap, generate_missing_values_chart

st.set_page_config(page_title="NeuroInsights", page_icon="📊", layout="wide")

st.title("📊 NeuroInsights — Automated Data Profiling & Visualization")
st.caption("Upload any CSV — get instant data profiling, auto-generated charts, and plain-English insights")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

use_sample = st.checkbox("Or use a sample dataset instead", value=not uploaded)

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample:
    import numpy as np
    np.random.seed(1)
    df = pd.DataFrame({
        "customer_id": range(1, 501),
        "age": np.random.normal(38, 12, 500).clip(18, 80).round(),
        "monthly_spend": np.random.exponential(200, 500).round(2),
        "city": np.random.choice(["Pune", "Mumbai", "Delhi", "Bangalore"], 500),
        "signup_date": pd.date_range("2023-01-01", periods=500, freq="D").astype(str),
        "is_active": np.random.choice([True, False], 500, p=[0.7, 0.3]),
    })
else:
    st.stop()

with st.spinner("Profiling dataset..."):
    profile = profile_dataframe(df)
    insights = generate_insights(profile)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{profile['shape']['rows']:,}")
c2.metric("Columns", profile['shape']['columns'])
c3.metric("Duplicate Rows", profile['duplicate_rows'])
c4.metric("Memory", f"{profile['memory_usage_kb']:.1f} KB")

st.subheader("🔍 Auto-Generated Insights")
for insight in insights:
    st.markdown(f"- {insight}")

st.subheader("📋 Column Profile")
profile_rows = []
for col, stats in profile["columns"].items():
    profile_rows.append({
        "Column": col, "Type": stats["type"], "Nulls %": stats["null_pct"],
        "Unique %": stats["unique_pct"],
        "Mean": stats.get("mean", "-"), "Std": stats.get("std", "-"),
    })
st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

st.subheader("📈 Auto-Generated Visualizations")
cols_to_plot = list(profile["columns"].keys())[:6]  # cap for performance
for i in range(0, len(cols_to_plot), 2):
    col_pair = cols_to_plot[i:i+2]
    cols = st.columns(len(col_pair))
    for j, col_name in enumerate(col_pair):
        col_type = profile["columns"][col_name]["type"]
        fig = generate_column_chart(df, col_name, col_type)
        if fig:
            cols[j].plotly_chart(fig, use_container_width=True)

numeric_cols = [c for c, s in profile["columns"].items() if s["type"] == "numeric"]
heatmap = generate_correlation_heatmap(df, numeric_cols)
if heatmap:
    st.subheader("🔗 Correlation Heatmap")
    st.plotly_chart(heatmap, use_container_width=True)

missing_chart = generate_missing_values_chart(profile)
if missing_chart:
    st.subheader("❓ Missing Values")
    st.plotly_chart(missing_chart, use_container_width=True)

st.caption("Built with Python, Pandas, Plotly, Streamlit | Rule-based profiling + insight generation, zero LLM dependency")
