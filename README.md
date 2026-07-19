# 🛢️ Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow Prediction

> A data-driven Vertical Lift Performance (VLP) model that predicts wellbore pressure drop
> using Random Forest regression, trained on real Volve field production data and benchmarked
> against the Beggs & Brill (1973) empirical correlation.

---

## Table of Contents

- [Abstract](#abstract)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Methodology Overview](#methodology-overview)
- [Validation Strategies](#validation-strategies)
- [Key Results](#key-results)
- [Feature Importance](#feature-importance)
- [RF vs Beggs & Brill](#rf-vs-beggs--brill)
- [Interactive Web App](#interactive-web-app)
- [File Descriptions](#file-descriptions)
- [Limitations & Caveats](#limitations--caveats)
- [Dependencies](#dependencies)
- [License](#license)

---

## Abstract

Vertical Lift Performance (VLP) models are essential in petroleum engineering for predicting the pressure drop across the wellbore during multiphase flow. Classical empirical correlations such as Beggs & Brill (1973) require detailed well geometry and fluid PVT data that may be unavailable or inaccurate. This project develops an alternative, data-driven approach using **Random Forest regression** trained on real production data from the **Volve field** (North Sea, Equinor).

The model predicts the tubing pressure drop **ΔP = P_wf − P_wh** (bottomhole pressure minus wellhead pressure) using 16 engineered features derived from daily production measurements. Three validation strategies are employed — Leave-One-Well-Out (LOWO) cross-validation, chronological within-well splits, and pooled random splits — to provide a complete and honest assessment of model generalization.

**Key result:** The Random Forest model achieves an **R² of 0.976** on pooled validation and substantially outperforms Beggs & Brill on every well, while requiring no assumed well geometry parameters.

---

## Quick Start

### Option A: Google Colab (Recommended — No Setup Required)

1. Download this repository (or clone it)
2. Open [Google Colab](https://colab.research.google.com/)
3. Upload `notebooks/01_Data_Analysis.ipynb` → click **Runtime → Run all**
4. Upload `notebooks/02_ML_Model.ipynb` → click **Runtime → Run all**
5. When prompted, upload `data/volve_welldata_raw.csv`
6. Download the generated figures and CSVs from the output cells

### Option B: Run Locally (Python Script)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling.git
cd Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete pipeline (produces all figures + CSVs)
python code/vlp_pipeline_v2.py
```

All outputs will be saved to `data/` (CSV results) and `figures/` (plots).

### Option C: Interactive Web App (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens a browser-based dashboard for exploring predictions, feature importance, and model performance interactively.

> 🚀 **To deploy online** (free via Streamlit Cloud), see the [Deployment Guide in SETUP.md](SETUP.md#deploying-to-streamlit-cloud-free-hosting).

> 📖 For detailed step-by-step instructions, troubleshooting, and prerequisites, see **[SETUP.md](SETUP.md)**.

---

## Repository Structure

```
Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling/
│
├── README.md                          ← This file — project overview and guide
├── SETUP.md                           ← Detailed step-by-step execution guide
├── METHODOLOGY.md                     ← Technical methodology documentation
├── RESULTS.md                         ← Full results summary with figures
├── PROJECT_LOG.md                     ← Honest record of corrections and caveats
├── requirements.txt                   ← Python dependencies (version-pinned)
├── LICENSE                            ← MIT License
├── .gitignore                         ← Git ignore rules
│
├── notebooks/
│   ├── 01_Data_Analysis.ipynb         ← Data exploration & feature engineering
│   └── 02_ML_Model.ipynb             ← Model training, validation & results
│
├── code/
│   ├── vlp_pipeline_v2.py            ← ★ Main pipeline — run this to reproduce all results
│   ├── vlp_pipeline_final.py         ← Original v1 pipeline (archived, for reference)
│   ├── diagnose_data.py              ← Data diagnostic utility (per-well statistics)
│   ├── create_notebooks.py           ← Script that programmatically generates notebooks
│   └── build_findings_pdf.py         ← PDF report generator
│
├── app/
│   └── streamlit_app.py              ← Interactive Streamlit web application
│
├── data/
│   ├── volve_welldata_raw.csv        ← Raw Volve production data (15,634 rows)
│   ├── modelready_features.csv       ← Cleaned, feature-engineered dataset (5,630 rows)
│   ├── lowo_results.csv              ← Leave-One-Well-Out validation metrics
│   ├── chrono_split_results.csv      ← Chronological split validation metrics
│   ├── pooled_split_results.csv      ← Pooled random split validation metrics
│   ├── validation_summary.csv        ← All 3 strategies compared side-by-side
│   ├── feature_importance_gini.csv   ← Gini-based feature importance rankings
│   ├── feature_importance_shap.csv   ← SHAP-based feature importance rankings
│   ├── beggs_brill_results.csv       ← Beggs & Brill benchmark results per well
│   ├── rf_vs_bb_comparison.csv       ← RF vs B&B side-by-side comparison
│   └── beggs_brill_comparison_ASSUMED_GEOMETRY.csv ← Full B&B predictions
│
└── figures/
    ├── fig1_lowo_predicted_vs_actual.png    ← LOWO scatter + per-fold R²/RMSE bars
    ├── fig2_chrono_split_timeseries.png     ← Chronological split time series
    ├── fig3_feature_importance_gini.png     ← Gini feature importance bar chart
    ├── fig4_shap_beeswarm.png              ← SHAP beeswarm plot
    ├── fig5_rf_vs_bb.png                   ← RF vs Beggs & Brill RMSE comparison
    ├── fig6_vlp_curve_by_wc.png            ← VLP curve colored by water cut
    ├── fig7_residual_analysis.png          ← 4-panel residual analysis
    └── fig8_validation_comparison.png      ← 3-strategy validation comparison
```

---

## Dataset

**Source:** [Volve Field Data (Equinor, 2018)](https://www.equinor.com/energy/volve-data-sharing) — a publicly released North Sea oil field dataset.

| Property | Value |
|----------|-------|
| Raw rows | 15,634 |
| Producer (OP) rows | 9,143 |
| Wells with PDG data | 5 (F-1C, F-11H, F-12H, F-14H, F-15D) |
| Model-ready rows (after filtering) | 5,630 |
| Time span | 2008–2016 |

**Key columns used from the raw data:**

| Column | Description | Unit |
|--------|-------------|------|
| `AVG_DOWNHOLE_PRESSURE` | Daily average bottomhole pressure (PDG) | bar |
| `AVG_WHP_P` | Daily average wellhead pressure | bar |
| `AVG_DP_TUBING` | Measured tubing pressure drop | bar |
| `BORE_OIL_VOL` | Daily oil production volume | Sm³/d |
| `BORE_GAS_VOL` | Daily gas production volume | Sm³/d |
| `BORE_WAT_VOL` | Daily water production volume | Sm³/d |
| `AVG_DOWNHOLE_TEMPERATURE` | Downhole temperature | °C |
| `AVG_WHT_P` | Wellhead temperature | °C |
| `AVG_CHOKE_SIZE_P` | Choke opening | % |
| `ON_STREAM_HRS` | Hours on production per day | hrs |

---

## Methodology Overview

The pipeline follows an 8-stage process:

```
Raw CSV → Filter → Feature Engineering → Steady-State Filter →
Hyperparameter Tuning → 3-Strategy Validation → Feature Importance →
Beggs & Brill Benchmark → Figures & Results
```

1. **Load & Filter** — Keep only producer wells with valid PDG + wellhead pressure data
2. **Feature Engineering** — Create 16 features: flow rates (normalized by on-stream hours), water cut, GOR, GLR, log-transformed rates, pressure ratio, temperature gradient
3. **Steady-State Filter** — 7-day rolling window removes transient data (CV < 0.35, σ(ΔP) < 30 bar)
4. **Hyperparameter Tuning** — GridSearchCV over `n_estimators`, `min_samples_leaf`, `max_features`
5. **Validation** — Three strategies (see below)
6. **Feature Importance** — Gini impurity + SHAP analysis
7. **Beggs & Brill Benchmark** — Classical correlation with assumed geometry
8. **Figures** — 8 publication-quality plots

> 📖 For full technical details, see **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Validation Strategies

Three validation strategies are used — not just the best-looking one:

| Strategy | What It Tests | How It Works | Strictness |
|----------|--------------|--------------|------------|
| **LOWO** (Leave-One-Well-Out) | Cross-well generalization | Train on 4 wells, test on the held-out 5th | ★★★ Very strict |
| **Chronological** (within-well) | Future prediction from past | Train on first 80% of each well's timeline | ★★☆ Moderate |
| **Pooled Random** (80/20) | Overall interpolation ability | Random 80/20 split across all data | ★☆☆ Least strict |

Reporting all three gives an honest picture: the model interpolates very well (pooled R²=0.976), but cross-well generalization varies significantly.

---

## Key Results

### Summary Metrics

| Strategy | Mean RMSE (bar) | Mean MAPE (%) | Mean R² |
|----------|:-----------:|:----------:|:-------:|
| LOWO (cross-well) | 14.98 | 7.30 | 0.170 |
| Chronological (within-well) | 9.86 | 4.14 | −2.28* |
| Pooled random (80/20) | **4.12** | **0.74** | **0.976** |

*\*Negative mean R² is driven by F-12H and F-14H where the chronological test set has a fundamentally different operating regime than the training period.*

### Per-Well LOWO Results

| Well | n_test | RMSE (bar) | MAPE (%) | R² |
|------|:------:|:----------:|:--------:|:--:|
| F-1C | 390 | 15.48 | 7.96 | 0.447 |
| F-11H | 1,063 | 7.76 | 3.24 | **0.862** |
| F-12H | 894 | 18.00 | 8.52 | −0.023 |
| F-14H | 2,553 | 15.42 | 6.62 | 0.663 |
| F-15D | 730 | 18.25 | 10.19 | −1.099 |

**Interpretation:** F-11H generalizes best (high-data, diverse conditions). F-15D and F-12H perform poorly cross-well due to unique operating regimes with narrow ΔP ranges that aren't represented in the training wells.

> 📖 For full results with figures, see **[RESULTS.md](RESULTS.md)**.

---

## Feature Importance

Top 5 features by Gini importance (trained on full dataset):

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | GLR (Gas-Liquid Ratio) | 0.410 |
| 2 | WC (Water Cut) | 0.258 |
| 3 | q_wat (Water Rate) | 0.111 |
| 4 | AVG_DOWNHOLE_TEMPERATURE | 0.050 |
| 5 | dT (Temperature Gradient) | 0.045 |

**Key insight:** Compositional features (GLR, WC, water rate) account for **~78%** of the model's predictive power. This makes physical sense — multiphase flow regime transitions are primarily driven by the gas-liquid ratio and water cut.

---

## RF vs Beggs & Brill

The Random Forest model outperforms the Beggs & Brill (1973) correlation on every well:

| Well | RF RMSE (bar) | B&B RMSE (bar) | RF Improvement |
|------|:------------:|:--------------:|:--------------:|
| F-1C | 15.48 | 67.94 | **77% lower** |
| F-11H | 7.76 | 61.87 | **87% lower** |
| F-12H | 18.00 | 53.88 | **67% lower** |
| F-14H | 15.42 | 64.53 | **76% lower** |
| F-15D | 18.25 | 69.11 | **74% lower** |

> ⚠️ **Important caveat:** The Beggs & Brill comparison uses **assumed** tubing geometry (ID=4.892 in, depth=3100 m, inclination=65°) because the raw CSV does not contain well completion data. The RF model does not depend on these assumptions.

---

## Interactive Web App

The Streamlit application provides:

- 🔮 **Real-time predictions** — input well conditions, get instant ΔP prediction
- 📊 **Model performance dashboard** — interactive LOWO results and metrics
- 📈 **Feature importance explorer** — Gini and SHAP visualizations
- 🔄 **What-if analysis** — explore how changing parameters affects predictions
- 📋 **Data explorer** — browse and filter the processed dataset

```bash
streamlit run app/streamlit_app.py
```

---

## File Descriptions

| File | Purpose | When to Use |
|------|---------|-------------|
| `code/vlp_pipeline_v2.py` | **Main pipeline** — reproduces all results from raw data | Run this to regenerate all CSVs and figures |
| `code/diagnose_data.py` | Per-well diagnostic statistics | Run to understand why certain wells perform poorly |
| `code/vlp_pipeline_final.py` | Original v1 pipeline (archived) | Reference only — superseded by v2 |
| `code/create_notebooks.py` | Generates the Jupyter notebooks | Only if you need to regenerate notebooks |
| `code/build_findings_pdf.py` | Generates the PDF findings report | Run after pipeline to create PDF |
| `app/streamlit_app.py` | Interactive web dashboard | Run with `streamlit run` for exploration |
| `notebooks/01_Data_Analysis.ipynb` | Data exploration notebook | Use in Colab/Jupyter for step-by-step analysis |
| `notebooks/02_ML_Model.ipynb` | Model training notebook | Use in Colab/Jupyter for step-by-step ML |

---

## Limitations & Caveats

1. **Assumed well geometry** — The Beggs & Brill benchmark uses assumed tubing ID, depth, and inclination (not measured values from completion reports). The RF model is unaffected.
2. **Single-field validation** — All data comes from the Volve field (North Sea). Model transferability to other fields is not established.
3. **Cross-well generalization varies** — LOWO R² ranges from −1.10 to +0.86 across the 5 wells, indicating the model struggles with wells having fundamentally different operating regimes.
4. **No real-time deployment testing** — The model was validated on historical data only.
5. **See `PROJECT_LOG.md`** for a complete, honest record of what was corrected from earlier drafts, including fabricated narratives and inconsistent numbers that were identified and fixed.

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.42.0
streamlit>=1.28.0
joblib>=1.2.0
```

Install all: `pip install -r requirements.txt`

**Python version:** 3.8 or higher recommended.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Dataset License

The Volve field dataset is provided by [Equinor (2018)](https://www.equinor.com/energy/volve-data-sharing) under the Equinor Open Data License.

---

*Last updated: July 2026*
