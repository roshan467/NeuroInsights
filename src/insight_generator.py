"""
insight_generator.py — converts profiling statistics into plain-English
insights using a rule-based system (transparent, explainable logic —
not an LLM, so there's zero hallucination risk on data facts).
"""

def generate_insights(profile: dict) -> list:
    insights = []
    shape = profile["shape"]
    insights.append(f"Dataset has {shape['rows']:,} rows and {shape['columns']} columns.")

    if profile["duplicate_rows"] > 0:
        pct = round(100 * profile["duplicate_rows"] / shape["rows"], 1) if shape["rows"] else 0
        insights.append(f"⚠️ Found {profile['duplicate_rows']} duplicate rows ({pct}% of data) — consider deduplication.")

    for col, stats in profile["columns"].items():
        if stats["null_pct"] > 30:
            insights.append(f"⚠️ Column '{col}' has {stats['null_pct']}% missing values — high enough to reconsider using it as-is.")
        elif stats["null_pct"] > 0:
            insights.append(f"Column '{col}' has {stats['null_pct']}% missing values.")

        if stats["type"] == "numeric":
            skew = stats.get("skewness", 0)
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append(f"Column '{col}' is significantly {direction}-skewed (skewness={skew}) — consider a log transform if used in modeling.")

            outliers = stats.get("outliers", {})
            if outliers.get("pct", 0) > 5:
                insights.append(f"⚠️ Column '{col}' has {outliers['pct']}% outliers (IQR method) — worth investigating before modeling.")

        if stats["type"] == "categorical" and stats["unique_count"] == 1:
            insights.append(f"Column '{col}' has only 1 unique value — likely not useful for analysis (zero variance).")

        if stats["unique_pct"] > 95 and stats["type"] != "numeric":
            insights.append(f"Column '{col}' is nearly all-unique ({stats['unique_pct']}%) — likely an ID column, not a feature.")

    for pair in profile.get("strong_correlations", [])[:5]:
        strength = "very strong" if abs(pair["correlation"]) > 0.8 else "strong"
        direction = "positive" if pair["correlation"] > 0 else "negative"
        insights.append(
            f"📊 {strength.capitalize()} {direction} correlation ({pair['correlation']}) between "
            f"'{pair['col1']}' and '{pair['col2']}'."
        )

    if not insights:
        insights.append("No significant data quality issues detected.")

    return insights
