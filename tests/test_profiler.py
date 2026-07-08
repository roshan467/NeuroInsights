"""
test_profiler.py — unit tests for the profiling and insight-generation
logic (pure data logic, no visualization dependency, fully deterministic).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from profiler import detect_column_type, detect_outliers_iqr, profile_dataframe
from insight_generator import generate_insights


def make_test_df():
    return pd.DataFrame({
        "id": range(1, 101),
        "age": np.concatenate([np.random.normal(35, 10, 95), [200, 210, -50, 220, 5]]),  # w/ outliers
        "category": np.random.choice(["A", "B", "C"], 100),
        "constant_col": ["X"] * 100,
        "price": np.random.normal(35, 10, 100),
    })


def test_detect_numeric_column():
    df = make_test_df()
    assert detect_column_type(df["age"]) == "numeric"


def test_detect_categorical_column():
    df = make_test_df()
    assert detect_column_type(df["category"]) == "categorical"


def test_detect_id_column_as_numeric_or_text():
    df = make_test_df()
    result = detect_column_type(df["id"])
    assert result in ("numeric", "text")


def test_outlier_detection_finds_injected_outliers():
    df = make_test_df()
    result = detect_outliers_iqr(df["age"])
    assert result["count"] >= 3  # we injected 4-5 extreme values


def test_profile_dataframe_shape():
    df = make_test_df()
    profile = profile_dataframe(df)
    assert profile["shape"]["rows"] == 100
    assert profile["shape"]["columns"] == 5


def test_profile_detects_zero_variance_column():
    df = make_test_df()
    profile = profile_dataframe(df)
    assert profile["columns"]["constant_col"]["unique_count"] == 1


def test_correlation_detected_between_correlated_columns():
    df = pd.DataFrame({"a": range(100), "b": [x * 2 + 1 for x in range(100)]})
    profile = profile_dataframe(df)
    assert len(profile["strong_correlations"]) >= 1
    assert profile["strong_correlations"][0]["correlation"] > 0.9


def test_insights_flag_zero_variance_column():
    df = make_test_df()
    profile = profile_dataframe(df)
    insights = generate_insights(profile)
    assert any("zero variance" in i.lower() for i in insights)


def test_insights_flag_outliers():
    df = make_test_df()
    profile = profile_dataframe(df)
    insights = generate_insights(profile)
    assert any("outlier" in i.lower() for i in insights)


if __name__ == "__main__":
    tests = [
        test_detect_numeric_column, test_detect_categorical_column, test_detect_id_column_as_numeric_or_text,
        test_outlier_detection_finds_injected_outliers, test_profile_dataframe_shape,
        test_profile_detects_zero_variance_column, test_correlation_detected_between_correlated_columns,
        test_insights_flag_zero_variance_column, test_insights_flag_outliers,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
