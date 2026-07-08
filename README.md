# NeuroInsights — Automated Data Profiling & Visualization Engine

Upload any CSV and get instant, automated data profiling, auto-selected visualizations, and plain-English insights — the core logic behind tools like pandas-profiling/Sweetviz, built from scratch to understand every step.

## What this demonstrates
- **Automated column-type detection** (numeric / categorical / datetime / text) using statistical heuristics, not hardcoding
- **Statistical profiling**: nulls, cardinality, skewness, IQR-based outlier detection, correlation analysis
- **Rule-based insight generation**: converts raw statistics into plain-English findings (e.g., "Column X is significantly right-skewed") — zero LLM dependency, so zero hallucination risk on data facts
- **Automatic chart-type selection**: histograms for numeric columns, bar charts for categorical, time-series for dates — picked automatically based on detected type
- Unit-tested profiling/insight logic (9/9 tests passing) — the deterministic, non-visual core is fully covered

## Architecture
```
NeuroInsights/
├── src/
│   ├── profiler.py           # column-type detection, stats, outliers, correlations
│   ├── insight_generator.py  # rule-based plain-English insight generation
│   ├── auto_visualizer.py    # automatic chart-type selection (Plotly)
│   └── app.py                # Streamlit app: upload → profile → visualize
├── tests/
│   └── test_profiler.py      # 9 tests, all passing
```

## How to run
```bash
pip install -r requirements.txt
streamlit run src/app.py
# Upload any CSV, or check "use a sample dataset" to try it immediately
```

## What makes the insights trustworthy
Unlike an LLM-generated summary, every insight here traces back to a specific, deterministic calculation (skewness formula, IQR bounds, correlation coefficient) — there's no hallucination risk on the facts. This is a deliberate design choice: **data profiling should never invent findings**, only surface real statistical properties.

## Honest note on scope
This is a general-purpose auto-EDA tool — it works on any tabular dataset (sales data, healthcare records, survey results, etc.), not a domain-specific one. If you want a version specialized for a particular domain (e.g., healthcare/EEG data specifically), that would need domain-specific feature extraction added on top of this general profiling engine.

## Tech stack
Python · Pandas · NumPy · Plotly · Streamlit
"# NeuroInsights" 
