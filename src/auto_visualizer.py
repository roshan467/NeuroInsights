"""
auto_visualizer.py — automatically selects and generates the right chart
type for each column based on its detected data type (the core logic
behind "upload any dataset, get instant visualizations" tools).
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def generate_column_chart(df: pd.DataFrame, col: str, col_type: str):
    """Returns a Plotly figure appropriate for the column's data type."""
    if col_type == "numeric":
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}",
                            marginal="box")
        return fig
    elif col_type == "categorical":
        counts = df[col].value_counts().head(15).reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Top values in {col}")
        return fig
    elif col_type == "datetime":
        try:
            dates = pd.to_datetime(df[col], errors="coerce")
            counts = dates.dt.to_period("M").value_counts().sort_index().reset_index()
            counts.columns = ["period", "count"]
            counts["period"] = counts["period"].astype(str)
            fig = px.line(counts, x="period", y="count", title=f"{col} over time", markers=True)
            return fig
        except Exception:
            return None
    return None


def generate_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation Heatmap")
    return fig


def generate_missing_values_chart(profile: dict):
    cols_with_nulls = {c: s["null_pct"] for c, s in profile["columns"].items() if s["null_pct"] > 0}
    if not cols_with_nulls:
        return None
    fig = px.bar(
        x=list(cols_with_nulls.keys()), y=list(cols_with_nulls.values()),
        labels={"x": "Column", "y": "% Missing"}, title="Missing Values by Column",
        color=list(cols_with_nulls.values()), color_continuous_scale="Reds",
    )
    return fig
