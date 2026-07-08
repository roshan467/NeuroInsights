"""
profiler.py — automated data profiling: dtype detection, null analysis,
statistical summaries, cardinality, skewness, and outlier detection.
This is the analysis engine behind tools like pandas-profiling/Sweetviz,
built from scratch to understand exactly what it's doing.
"""
import pandas as pd
import numpy as np


def detect_column_type(series: pd.Series) -> str:
    """Classifies a column as numeric, categorical, datetime, or boolean."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # try parsing as datetime
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(series.dropna().head(20), errors="raise")
        return "datetime"
    except (ValueError, TypeError):
        pass
    n_unique = series.nunique()
    if n_unique <= max(20, len(series) * 0.05):
        return "categorical"
    return "text"


def detect_outliers_iqr(series: pd.Series):
    """IQR-based outlier detection (a real, standard statistical technique)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "count": int(len(outliers)),
        "pct": round(100 * len(outliers) / len(series), 2) if len(series) else 0,
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }

    for col in df.columns:
        series = df[col]
        col_type = detect_column_type(series)
        col_profile = {
            "type": col_type,
            "null_count": int(series.isnull().sum()),
            "null_pct": round(100 * series.isnull().sum() / len(series), 2) if len(series) else 0,
            "unique_count": int(series.nunique()),
            "unique_pct": round(100 * series.nunique() / len(series), 2) if len(series) else 0,
        }

        if col_type == "numeric":
            clean = series.dropna()
            col_profile.update({
                "mean": round(float(clean.mean()), 4) if len(clean) else None,
                "median": round(float(clean.median()), 4) if len(clean) else None,
                "std": round(float(clean.std()), 4) if len(clean) else None,
                "min": round(float(clean.min()), 4) if len(clean) else None,
                "max": round(float(clean.max()), 4) if len(clean) else None,
                "skewness": round(float(clean.skew()), 4) if len(clean) > 2 else 0,
                "outliers": detect_outliers_iqr(clean) if len(clean) > 4 else {"count": 0, "pct": 0},
            })
        elif col_type == "categorical":
            top_values = series.value_counts().head(5).to_dict()
            col_profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        profile["columns"][col] = col_profile

    # correlation matrix for numeric columns
    numeric_cols = [c for c, p in profile["columns"].items() if p["type"] == "numeric"]
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        strong_pairs = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = corr.loc[c1, c2]
                if abs(val) >= 0.5:
                    strong_pairs.append({"col1": c1, "col2": c2, "correlation": float(val)})
        profile["strong_correlations"] = sorted(strong_pairs, key=lambda x: -abs(x["correlation"]))

    return profile
