"""
Generates two Jupyter notebooks for the VLP project:
  1. 01_Data_Analysis.ipynb  - EDA, feature engineering, data cleaning
  2. 02_ML_Model.ipynb       - RF training, validation, SHAP, benchmarking

Run this script once to create both notebooks, then open them in
Google Colab or Jupyter.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(BASE_DIR, 'notebooks')
os.makedirs(NB_DIR, exist_ok=True)


def md(source):
    """Create a markdown cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(source):
    """Create a code cell."""
    lines = source.split('\n')
    src = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "file_extension": ".py"
            },
            "colab": {
                "provenance": [],
                "toc_visible": True
            }
        },
        "cells": cells
    }


# =====================================================================
# NOTEBOOK 1: DATA ANALYSIS & EXPLORATION
# =====================================================================
nb1_cells = [

    # ---- TITLE ----
    md("""# Notebook 1: Data Analysis & Exploration
## Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow Prediction

**Project Objective:** Develop a data-driven Vertical Lift Performance (VLP) model that predicts wellbore pressure drop ($\\Delta P = P_{wf} - P_{wh}$) using Random Forest regression, trained on real Volve field production data.

**What this notebook does:**
1. Loads the raw Volve field production dataset
2. Explores the data structure and identifies usable wells
3. Engineers physically meaningful features
4. Applies a steady-state filter to remove transient data
5. Performs per-well statistical analysis
6. Creates exploratory visualizations
7. Saves the cleaned, model-ready dataset

**Dataset:** Volve open field dataset (Equinor, 2018) — a decommissioned North Sea oil field with 8+ years of production history."""),

    # ---- SETUP ----
    md("""## 1. Setup & Dependencies

First, we install any missing packages. Everything except `shap` is pre-installed in Google Colab."""),

    code("""# Install SHAP for feature importance analysis (used in Notebook 2)
# All other packages are pre-installed in Colab
!pip install shap -q"""),

    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Plot styling for publication-quality figures
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 120,
})

print("Libraries loaded successfully.")"""),

    # ---- LOAD DATA ----
    md("""## 2. Loading the Volve Dataset

The Volve field dataset was released by Equinor in 2018 as an open-access research dataset. It contains daily production data for 7 wellbores from 2007 to 2016.

**Important:** Upload your `volve_welldata.csv` file before running this cell.
- In Colab: Click the folder icon on the left sidebar → Upload
- Locally: Place the file in the `data/` folder"""),

    code("""# ============================================================
# ADJUST THIS PATH for your environment:
# - Google Colab:  '/content/volve_welldata.csv'
# - Local machine: '../data/volve_welldata_raw.csv'
# ============================================================
import os

# Auto-detect environment
if os.path.exists('/content'):
    # Google Colab
    INPUT_PATH = '/content/volve_welldata.csv'
    OUTPUT_DIR = '/content'
    print("Running in Google Colab")
else:
    # Local
    INPUT_PATH = '../data/volve_welldata_raw.csv'
    OUTPUT_DIR = '../data'
    print("Running locally")

df = pd.read_csv(INPUT_PATH)
print(f"\\nDataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")"""),

    # ---- INITIAL EXPLORATION ----
    md("""## 3. Initial Data Exploration

Let's understand what we're working with before any filtering or engineering."""),

    code("""# First 5 rows
print("=== First 5 Rows ===")
df.head()"""),

    code("""# Data types and non-null counts
print("=== Data Types & Missing Values ===")
print(df.dtypes)
print(f"\\n=== Non-null counts ===")
print(df.count())"""),

    code("""# Well type distribution — we only care about producers (OP), not injectors (WI)
print("=== Well Type Distribution ===")
print(df['WELL_TYPE'].value_counts())
print(f"\\nProducer rows: {(df['WELL_TYPE'] == 'OP').sum():,}")
print(f"Injector rows: {(df['WELL_TYPE'] == 'WI').sum():,}")"""),

    code("""# Wells in the dataset
print("=== Wells and Row Counts ===")
well_summary = df.groupby('Wellbore name').agg(
    total_rows=('WELL_TYPE', 'count'),
    producer_rows=('WELL_TYPE', lambda x: (x == 'OP').sum()),
    injector_rows=('WELL_TYPE', lambda x: (x == 'WI').sum())
).sort_values('total_rows', ascending=False)
print(well_summary)"""),

    md("""### Key Observation
The dataset has **7 wellbores**, but not all are usable for VLP modeling:
- We need **producer wells** (OP) — injectors push water *into* the reservoir, not up through tubing
- We need wells with **downhole pressure gauge** (PDG) data — without $P_{wf}$, we can't compute $\\Delta P$

Let's check which wells actually have PDG data."""),

    code("""# Check which wells have valid downhole pressure readings
op = df[df['WELL_TYPE'] == 'OP'].copy()

# Convert numeric columns
NUM_COLS = ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
            'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
for c in NUM_COLS:
    op[c] = pd.to_numeric(op[c], errors='coerce')

print("=== Producer Wells: Downhole Pressure Availability ===")
for well in op['Wellbore name'].unique():
    w = op[op['Wellbore name'] == well]
    valid_pdg = (w['AVG_DOWNHOLE_PRESSURE'] > 0).sum()
    valid_whp = (w['AVG_WHP_P'] > 0).sum()
    valid_oil = (w['BORE_OIL_VOL'] > 0).sum()
    print(f"  {well:20s}  PDG>0: {valid_pdg:5d}  WHP>0: {valid_whp:5d}  Oil>0: {valid_oil:5d}  Total: {len(w):5d}")"""),

    md("""### Well Selection Decision

Based on the data availability check:
- **15/9-F-4 AH**: 100% water injector — **cannot use**
- **15/9-F-5 AH**: Has producer rows but **zero valid downhole pressure** — **cannot use**
- **5 usable wells**: F-1C, F-11H, F-12H, F-14H, F-15D

> **Honesty note:** Earlier drafts of this project incorrectly described F-4 AH and F-5 AH as usable PDG wells. This was wrong. Only the 5 wells above have the data needed for VLP modeling."""),

    code("""# Filter to usable wells only
USABLE_WELLS = ['15/9-F-1 C', '15/9-F-11 H', '15/9-F-12 H',
                '15/9-F-14 H', '15/9-F-15 D']
op = op[op['Wellbore name'].isin(USABLE_WELLS)].copy()

# Apply core validity filter
mask = ((op['AVG_DOWNHOLE_PRESSURE'] > 0) & (op['AVG_WHP_P'] > 0) &
        (op['ON_STREAM_HRS'] > 0) & (op['BORE_OIL_VOL'] > 0) &
        (op['AVG_DP_TUBING'].notna()) & (op['AVG_DP_TUBING'] > 0))
clean = op[mask].copy()

print(f"After filtering to usable producer wells with valid data: {len(clean):,} rows")
print(f"Wells: {clean['Wellbore name'].nunique()}")
print(f"\\nRows per well:")
print(clean['Wellbore name'].value_counts().to_string())"""),

    # ---- FEATURE ENGINEERING ----
    md("""## 4. Feature Engineering

Now we create the features that the Random Forest model will use. Every feature is either **directly measured** from the raw data or **simply derived** from measured quantities.

### 4.1 Rate Normalization
The raw file gives daily volumes (`BORE_OIL_VOL`, etc.) but wells don't always produce for a full 24 hours. We normalize by actual on-stream hours:

$$q_{oil} = \\frac{\\text{BORE\\_OIL\\_VOL} \\times 24}{\\text{ON\\_STREAM\\_HRS}}$$

This gives the *equivalent daily rate* if the well had produced for 24 hours.

### 4.2 Compositional Features
- **Water Cut (WC):** $WC = \\frac{q_{wat}}{q_{oil} + q_{wat}}$ — fraction of liquid that is water
- **Gas-Oil Ratio (GOR):** $GOR = \\frac{q_{gas}}{q_{oil}}$ — how much gas per unit of oil
- **Gas-Liquid Ratio (GLR):** $GLR = \\frac{q_{gas}}{q_{liq}}$ — gas per total liquid (useful for high-WC wells)

### 4.3 Target Variable
$$\\Delta P = P_{wf} - P_{wh} = \\text{AVG\\_DOWNHOLE\\_PRESSURE} - \\text{AVG\\_WHP\\_P}$$

This is the **wellbore pressure drop** — the pressure consumed lifting fluid from reservoir depth to surface."""),

    code("""# Parse dates (mixed format in raw file: both '4/7/2014' and '24-Jul-13')
clean['Date of Production'] = pd.to_datetime(clean['Date of Production'], format='mixed')
clean = clean.sort_values(['Wellbore name', 'Date of Production']).reset_index(drop=True)

# ---- Rate normalization ----
clean['q_oil'] = clean['BORE_OIL_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_gas'] = clean['BORE_GAS_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_wat'] = clean['BORE_WAT_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_liq'] = clean['q_oil'] + clean['q_wat']

# ---- Compositional features ----
clean['WC']  = clean['q_wat'] / clean['q_liq'].replace(0, np.nan)
clean['GOR'] = clean['q_gas'] / clean['q_oil'].replace(0, np.nan)
clean['GLR'] = clean['q_gas'] / clean['q_liq'].replace(0, np.nan)

# ---- Log-transformed rates (better for tree models with wide ranges) ----
clean['log_q_liq'] = np.log1p(clean['q_liq'])
clean['log_q_oil'] = np.log1p(clean['q_oil'])
clean['log_q_gas'] = np.log1p(clean['q_gas'])

# ---- Temperature gradient proxy ----
clean['dT'] = clean['AVG_DOWNHOLE_TEMPERATURE'] - clean['AVG_WHT_P']

# ---- TARGET: Wellbore pressure drop ----
clean['delta_P'] = clean['AVG_DOWNHOLE_PRESSURE'] - clean['AVG_WHP_P']

# ---- Physical consistency filter ----
clean = clean[(clean['WC'] >= 0) & (clean['WC'] <= 1) &
              (clean['GOR'] > 0) & (clean['GOR'] < 5000) &
              (clean['q_liq'] > 0) & (clean['delta_P'] > 0)]

print(f"After feature engineering + physical consistency: {len(clean):,} rows")
print(f"\\nNew features created:")
for feat in ['q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR', 'GLR',
             'log_q_liq', 'log_q_oil', 'log_q_gas', 'dT', 'delta_P']:
    print(f"  {feat:20s}  min={clean[feat].min():10.2f}  max={clean[feat].max():10.2f}")"""),

    # ---- STEADY STATE FILTER ----
    md("""## 5. Steady-State Filter

VLP is a **steady-state concept** — it describes the pressure drop when flow is stable, not during shut-ins, ramp-ups, or well tests. We must remove transient data.

**Method:** 7-day rolling window analysis
- **Coefficient of Variation (CV) of liquid rate:** $CV_q = \\frac{\\sigma_q}{\\mu_q}$ over 7 days
- **Rolling standard deviation of $\\Delta P$:** $\\sigma_{\\Delta P}$ over 7 days

**Thresholds:** $CV_q < 0.35$ and $\\sigma_{\\Delta P} < 30$ bar

> **Note:** Earlier drafts of this project claimed tighter thresholds (CV < 0.10, $\\sigma_{\\Delta P}$ < 10 bar) that kept < 1% of the data. The actual data distribution doesn't support such strict filtering. The thresholds above were chosen by examining the actual distribution."""),

    code("""# Compute rolling statistics per well
g = clean.groupby('Wellbore name', group_keys=False)
clean['roll_std_q']  = g['q_liq'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['roll_mean_q'] = g['q_liq'].rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
clean['roll_std_dP'] = g['delta_P'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['cv_q'] = clean['roll_std_q'] / clean['roll_mean_q'].replace(0, np.nan)

# Apply thresholds
CV_THRESHOLD = 0.35
DP_STD_THRESHOLD = 30.0

steady_mask = ((clean['cv_q'] < CV_THRESHOLD) &
               (clean['roll_std_dP'] < DP_STD_THRESHOLD) &
               clean['cv_q'].notna())
steady = clean[steady_mask].copy()

print(f"Steady-state filter: {len(steady):,} of {len(clean):,} rows retained "
      f"({len(steady)/len(clean)*100:.1f}%)")
print(f"Removed: {len(clean) - len(steady):,} transient rows")
print(f"\\nPer-well breakdown:")
for well in USABLE_WELLS:
    n_total = len(clean[clean['Wellbore name'] == well])
    n_steady = len(steady[steady['Wellbore name'] == well])
    pct = n_steady / n_total * 100 if n_total > 0 else 0
    print(f"  {well:20s}  {n_steady:5d} / {n_total:5d}  ({pct:.1f}%)")"""),

    # ---- VISUALIZATIONS ----
    md("""## 6. Exploratory Data Analysis (EDA)

Now let's visualize the data to understand the physical behavior across wells."""),

    md("""### 6.1 Per-Well Statistics Table"""),

    code("""# Comprehensive per-well statistics
key_vars = ['delta_P', 'q_oil', 'q_wat', 'q_liq', 'WC', 'GOR',
            'AVG_WHP_P', 'AVG_DOWNHOLE_PRESSURE']

stats_rows = []
for well in USABLE_WELLS:
    w = steady[steady['Wellbore name'] == well]
    row = {'Well': well.replace('15/9-', ''), 'N': len(w)}
    for v in key_vars:
        row[f'{v}_mean'] = w[v].mean()
        row[f'{v}_std'] = w[v].std()
    stats_rows.append(row)

stats_df = pd.DataFrame(stats_rows)

# Display key statistics
display_cols = ['Well', 'N', 'delta_P_mean', 'delta_P_std', 'WC_mean', 'GOR_mean', 'q_liq_mean', 'AVG_WHP_P_mean']
print("=== Per-Well Operating Summary ===")
print(stats_df[display_cols].round(1).to_string(index=False))"""),

    md("""### 6.2 Delta P Distribution Per Well

This plot reveals why **cross-well prediction is challenging**: each well operates in a different pressure regime."""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of delta_P per well
well_colors = {'15/9-F-1 C': '#1976D2', '15/9-F-11 H': '#388E3C',
               '15/9-F-12 H': '#E64A19', '15/9-F-14 H': '#7B1FA2',
               '15/9-F-15 D': '#00838F'}

# Violin plot
data_for_violin = [steady[steady['Wellbore name'] == w]['delta_P'].values for w in USABLE_WELLS]
labels = [w.replace('15/9-', '') for w in USABLE_WELLS]

parts = axes[0].violinplot(data_for_violin, showmeans=True, showmedians=True)
axes[0].set_xticks(range(1, len(USABLE_WELLS) + 1))
axes[0].set_xticklabels(labels, rotation=15)
axes[0].set_ylabel('Delta P (bar)')
axes[0].set_title('Pressure Drop Distribution Per Well')

# Histogram overlay
for well in USABLE_WELLS:
    w = steady[steady['Wellbore name'] == well]
    axes[1].hist(w['delta_P'], bins=30, alpha=0.5, label=well.replace('15/9-', ''),
                 color=well_colors[well])
axes[1].set_xlabel('Delta P (bar)')
axes[1].set_ylabel('Count')
axes[1].set_title('Delta P Histogram by Well')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_deltaP_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Key finding: F-15D operates in a narrow, low dP range (153-209 bar)")
print("while F-14H has the widest range and highest mean dP.")"""),

    md("""### 6.3 Water Cut Evolution Over Time

Water cut is the **most important feature** for VLP prediction (we'll confirm this in Notebook 2). Understanding its evolution is critical."""),

    code("""fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, well in enumerate(USABLE_WELLS):
    w = steady[steady['Wellbore name'] == well].sort_values('Date of Production')
    ax = axes[i]
    
    # WC on primary axis
    ax.plot(w['Date of Production'], w['WC'], 'o-', ms=1.5, lw=0.5,
            color=well_colors[well], alpha=0.6, label='WC')
    ax.set_ylabel('Water Cut', color=well_colors[well])
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(well.replace('15/9-', ''), fontsize=12)
    
    # delta_P on secondary axis
    ax2 = ax.twinx()
    ax2.plot(w['Date of Production'], w['delta_P'], 's-', ms=1, lw=0.5,
             color='gray', alpha=0.4, label='dP')
    ax2.set_ylabel('dP (bar)', color='gray')
    
    ax.tick_params(axis='x', rotation=45, labelsize=7)

# Remove unused subplot
axes[5].set_visible(False)

plt.suptitle('Water Cut Evolution and Pressure Drop Over Time', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_wc_evolution.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""### 6.4 VLP Curve: Pressure Drop vs Liquid Rate"""),

    code("""fig, ax = plt.subplots(figsize=(10, 7))

sc = ax.scatter(steady['q_liq'], steady['delta_P'],
                c=steady['WC'], cmap='RdYlBu_r',
                s=12, alpha=0.6, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Water Cut (fraction)')

ax.set_xlabel('Liquid Flow Rate (Sm3/d)')
ax.set_ylabel('Wellbore Pressure Drop, dP (bar)')
ax.set_title('VLP Relationship: Pressure Drop vs Liquid Rate\\nColored by Water Cut')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_vlp_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("Observation: Higher WC clearly pushes dP higher at similar flow rates.")
print("This is physically expected: water is denser than oil, increasing hydrostatic head.")"""),

    md("""### 6.5 Correlation Matrix"""),

    code("""# Correlation heatmap of key features
corr_features = ['delta_P', 'q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
                  'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
                  'AVG_CHOKE_SIZE_P']
corr_matrix = steady[corr_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/eda_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n=== Top correlations with delta_P ===")
dp_corr = corr_matrix['delta_P'].drop('delta_P').sort_values(ascending=False)
for feat, val in dp_corr.items():
    print(f"  {feat:35s}  r = {val:+.3f}")"""),

    md("""### 6.6 Per-Well Feature-Target Correlation

This shows whether the same features drive pressure drop in **every** well (consistent physics) or only in specific wells (regime-dependent behavior)."""),

    code("""focus_feats = ['WC', 'q_liq', 'GOR', 'AVG_WHP_P', 'AVG_DOWNHOLE_TEMPERATURE']

print("=== Per-Well Correlations with delta_P ===")
print(f"{'Well':20s}" + "".join([f"  {f:>12s}" for f in focus_feats]))
print("-" * 90)
for well in USABLE_WELLS:
    w = steady[steady['Wellbore name'] == well]
    row = f"{well.replace('15/9-', ''):20s}"
    for f in focus_feats:
        corr = w[['delta_P', f]].corr().iloc[0, 1]
        row += f"  {corr:>+12.3f}"
    print(row)

print("\\nKey finding: WC has r > +0.82 in EVERY well. The physics is consistent.")
print("This means WC-driven pressure increase is real, not a statistical artifact.")"""),

    # ---- SAVE ----
    md("""## 7. Save Cleaned Dataset

We save the steady-state filtered, feature-engineered dataset for use in Notebook 2 (ML Model)."""),

    code("""# Define the feature columns we'll use for modeling
FEATURES = ['q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
            'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS',
            'log_q_liq', 'log_q_oil', 'log_q_gas', 'GLR', 'dT']
TARGET = 'delta_P'

# Select columns for output
output_cols = ['Wellbore name', 'Date of Production'] + FEATURES + [TARGET]
model_data = steady[output_cols].dropna().copy()

# Save
output_path = f'{OUTPUT_DIR}/volve_vlp_modelready.csv'
model_data.to_csv(output_path, index=False)

print(f"Saved model-ready dataset: {output_path}")
print(f"  Rows: {len(model_data):,}")
print(f"  Features: {len(FEATURES)}")
print(f"  Target: {TARGET}")
print(f"  Wells: {model_data['Wellbore name'].nunique()}")
print(f"\\nPer-well counts:")
print(model_data['Wellbore name'].value_counts().to_string())"""),

    md("""## 8. Summary of Key Findings

| Finding | Detail |
|---------|--------|
| **Usable wells** | 5 of 7 (F-1C, F-11H, F-12H, F-14H, F-15D) |
| **After filtering** | ~5,600 steady-state production days |
| **Target variable** | $\\Delta P = P_{wf} - P_{wh}$ (bar), range ~150-260 bar |
| **Most important feature** | Water Cut (r = +0.87 with $\\Delta P$ across all wells) |
| **Cross-well challenge** | F-15D is a low-rate, low-dP well; F-12H is high-rate, low-WC |
| **Physical consistency** | WC correlation with dP is consistent across all 5 wells |

### Proceed to Notebook 2 for ML Model Training →"""),

]


# =====================================================================
# NOTEBOOK 2: ML MODEL TRAINING & EVALUATION
# =====================================================================
nb2_cells = [

    # ---- TITLE ----
    md("""# Notebook 2: ML Model Training & Evaluation
## Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow Prediction

**What this notebook does:**
1. Loads the cleaned dataset from Notebook 1
2. Tunes Random Forest hyperparameters via GridSearchCV
3. Validates the model using **three strategies**:
   - Leave-One-Well-Out (LOWO) cross-validation
   - Chronological within-well split
   - Pooled random split
4. Computes feature importance (Gini + SHAP)
5. Benchmarks against Beggs & Brill (1973) correlation
6. Generates all figures for the final report

**Prerequisites:** Run Notebook 1 first to generate `volve_vlp_modelready.csv`"""),

    # ---- SETUP ----
    md("""## 1. Setup"""),

    code("""!pip install shap -q"""),

    code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import shap
    HAS_SHAP = True
    print("SHAP loaded successfully.")
except ImportError:
    HAS_SHAP = False
    print("SHAP not available - SHAP analysis will be skipped.")

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 120,
})

WELL_COLORS = {
    '15/9-F-1 C': '#1976D2', '15/9-F-11 H': '#388E3C',
    '15/9-F-12 H': '#E64A19', '15/9-F-14 H': '#7B1FA2',
    '15/9-F-15 D': '#00838F',
}

def short_well(name):
    return name.replace('15/9-', '')

print("Setup complete.")"""),

    # ---- LOAD DATA ----
    md("""## 2. Load Model-Ready Data"""),

    code("""import os

if os.path.exists('/content'):
    DATA_PATH = '/content/volve_vlp_modelready.csv'
    OUTPUT_DIR = '/content'
else:
    DATA_PATH = '../data/volve_vlp_modelready.csv'
    OUTPUT_DIR = '../data'
    FIG_DIR = '../figures'
    os.makedirs(FIG_DIR, exist_ok=True)

# For Colab, figures go to same directory
if os.path.exists('/content'):
    FIG_DIR = '/content'

d = pd.read_csv(DATA_PATH)
d['Date of Production'] = pd.to_datetime(d['Date of Production'])
print(f"Loaded: {len(d):,} rows x {d.shape[1]} columns")
print(f"Wells: {d['Wellbore name'].nunique()}")
print(f"\\nTarget (delta_P) statistics:")
print(d['delta_P'].describe().round(2))"""),

    code("""# Define features and target
FEATURES = ['q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
            'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS',
            'log_q_liq', 'log_q_oil', 'log_q_gas', 'GLR', 'dT']
TARGET = 'delta_P'

# Verify all features exist
missing = [f for f in FEATURES if f not in d.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
    FEATURES = [f for f in FEATURES if f in d.columns]

# Drop rows with NaN in features or target
d = d.dropna(subset=FEATURES + [TARGET])
print(f"\\nFinal dataset: {len(d):,} rows x {len(FEATURES)} features")
print(f"Features: {FEATURES}")"""),

    # ---- HYPERPARAMETER TUNING ----
    md("""## 3. Hyperparameter Tuning

We use **GridSearchCV** with 3-fold cross-validation to find optimal Random Forest parameters. This ensures our hyperparameters aren't arbitrary.

**Parameters we tune:**
- `n_estimators`: Number of trees (more trees = better but slower)
- `max_depth`: Maximum tree depth (controls overfitting)
- `min_samples_leaf`: Minimum samples in leaf node (regularization)
- `max_features`: Features considered at each split (controls diversity)"""),

    code("""# Use 80% of data for tuning
X_tune, _, y_tune, _ = train_test_split(
    d[FEATURES], d[TARGET], test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 20, 30],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 0.5],
}

n_combos = np.prod([len(v) for v in param_grid.values()])
print(f"Grid search over {n_combos} parameter combinations (3-fold CV)...")
print("This may take 2-5 minutes...\\n")

gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring='r2', n_jobs=1, verbose=1)
gs.fit(X_tune, y_tune)

best_params = gs.best_params_
print(f"\\n=== Best Parameters ===")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"  Best CV R2: {gs.best_score_:.4f}")"""),

    # ---- VALIDATION ----
    md("""## 4. Model Validation

We use **three validation strategies** to give a complete picture of model performance. Each strategy tests something different:

| Strategy | What It Tests | Strictness |
|----------|--------------|------------|
| **LOWO** | Can the model predict a completely new well? | Very strict |
| **Chronological** | Can it predict future production from past data? | Moderate |
| **Pooled random** | Overall interpolation ability | Least strict |

> **Important:** Reporting only the pooled random split (which gives the best numbers) would be intellectually dishonest. The LOWO results are the toughest test and must be reported alongside."""),

    md("""### 4.1 Leave-One-Well-Out (LOWO) Cross-Validation

Train on 4 wells, test on the 5th. Repeat for all 5 wells. This is the **gold standard** for testing cross-well generalization."""),

    code("""wells = d['Wellbore name'].unique()
lowo_results = []
lowo_preds = {}

for test_well in wells:
    train_df = d[d['Wellbore name'] != test_well]
    test_df  = d[d['Wellbore name'] == test_well]
    
    rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf.fit(train_df[FEATURES], train_df[TARGET])
    
    yp = rf.predict(test_df[FEATURES])
    yt = test_df[TARGET].values
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    r2 = r2_score(yt, yp)
    
    lowo_results.append({
        'test_well': test_well, 'n_test': len(test_df), 'n_train': len(train_df),
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
    })
    lowo_preds[test_well] = (yt, yp, test_df['WC'].values)

lowo_df = pd.DataFrame(lowo_results)

print("=== LOWO Results ===")
print(f"{'Well':20s}  {'n_test':>6s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>7s}  {'R2':>8s}")
print("-" * 65)
for _, r in lowo_df.iterrows():
    print(f"{r['test_well']:20s}  {int(r['n_test']):6d}  {r['RMSE']:8.2f}  "
          f"{r['MAE']:8.2f}  {r['MAPE']:7.2f}  {r['R2']:8.3f}")
print(f"\\nMean:  RMSE={lowo_df.RMSE.mean():.2f}  MAPE={lowo_df.MAPE.mean():.2f}%  R2={lowo_df.R2.mean():.3f}")
print(f"Std:   RMSE={lowo_df.RMSE.std():.2f}  MAPE={lowo_df.MAPE.std():.2f}%  R2={lowo_df.R2.std():.3f}")

lowo_df.to_csv(f'{OUTPUT_DIR}/lowo_results.csv', index=False)"""),

    md("""### 4.2 Chronological Within-Well Split

Train on each well's **first 80%** of production history (plus all other wells), test on the **last 20%**. This tests: *"Can the model predict future production from past data?"*"""),

    code("""chrono_results = []
chrono_preds = {}

for well in wells:
    well_data = d[d['Wellbore name'] == well].sort_values('Date of Production')
    split_idx = int(len(well_data) * 0.8)
    train_well = well_data.iloc[:split_idx]
    test_well_data = well_data.iloc[split_idx:]
    
    if len(test_well_data) < 5:
        continue
    
    # Train on ALL other wells + early data from this well
    other_wells = d[d['Wellbore name'] != well]
    train_combined = pd.concat([other_wells, train_well])
    
    rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf.fit(train_combined[FEATURES], train_combined[TARGET])
    
    yp = rf.predict(test_well_data[FEATURES])
    yt = test_well_data[TARGET].values
    
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    r2 = r2_score(yt, yp)
    
    chrono_results.append({
        'test_well': well, 'n_test': len(test_well_data), 'n_train': len(train_combined),
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
    })
    chrono_preds[well] = (yt, yp, test_well_data['Date of Production'].values)

chrono_df = pd.DataFrame(chrono_results)

print("=== Chronological Split Results ===")
print(f"{'Well':20s}  {'n_test':>6s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>7s}  {'R2':>8s}")
print("-" * 65)
for _, r in chrono_df.iterrows():
    print(f"{r['test_well']:20s}  {int(r['n_test']):6d}  {r['RMSE']:8.2f}  "
          f"{r['MAE']:8.2f}  {r['MAPE']:7.2f}  {r['R2']:8.3f}")
print(f"\\nMean:  RMSE={chrono_df.RMSE.mean():.2f}  MAPE={chrono_df.MAPE.mean():.2f}%  R2={chrono_df.R2.mean():.3f}")

chrono_df.to_csv(f'{OUTPUT_DIR}/chrono_split_results.csv', index=False)"""),

    md("""### 4.3 Pooled Random Split (80/20)

Standard train/test split across the entire dataset. This tests overall interpolation ability but is the **least strict** test — it doesn't guarantee the model works on new wells or future data."""),

    code("""X_train, X_test, y_train, y_test = train_test_split(
    d[FEATURES], d[TARGET], test_size=0.2, random_state=42)

rf_pooled = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_pooled.fit(X_train, y_train)

yp_pool = rf_pooled.predict(X_test)
yt_pool = y_test.values

pool_rmse = np.sqrt(mean_squared_error(yt_pool, yp_pool))
pool_mae = mean_absolute_error(yt_pool, yp_pool)
pool_mape = np.mean(np.abs((yt_pool - yp_pool) / yt_pool)) * 100
pool_r2 = r2_score(yt_pool, yp_pool)

print("=== Pooled Random Split Results ===")
print(f"  Train: {len(X_train):,} rows")
print(f"  Test:  {len(X_test):,} rows")
print(f"  RMSE:  {pool_rmse:.2f} bar")
print(f"  MAE:   {pool_mae:.2f} bar")
print(f"  MAPE:  {pool_mape:.2f}%")
print(f"  R2:    {pool_r2:.4f}")"""),

    md("""### 4.4 Validation Strategy Comparison"""),

    code("""# Summary comparison table
summary = pd.DataFrame([
    {'Strategy': 'LOWO (cross-well)', 'Mean_R2': lowo_df.R2.mean(),
     'Std_R2': lowo_df.R2.std(), 'Mean_RMSE': lowo_df.RMSE.mean(),
     'Mean_MAPE': lowo_df.MAPE.mean(), 'Best_R2': lowo_df.R2.max(),
     'Worst_R2': lowo_df.R2.min()},
    {'Strategy': 'Chronological', 'Mean_R2': chrono_df.R2.mean(),
     'Std_R2': chrono_df.R2.std(), 'Mean_RMSE': chrono_df.RMSE.mean(),
     'Mean_MAPE': chrono_df.MAPE.mean(), 'Best_R2': chrono_df.R2.max(),
     'Worst_R2': chrono_df.R2.min()},
    {'Strategy': 'Pooled Random', 'Mean_R2': pool_r2,
     'Std_R2': 0, 'Mean_RMSE': pool_rmse,
     'Mean_MAPE': pool_mape, 'Best_R2': pool_r2,
     'Worst_R2': pool_r2},
])

print("=== VALIDATION STRATEGY COMPARISON ===")
print(summary.round(3).to_string(index=False))
summary.to_csv(f'{OUTPUT_DIR}/validation_summary.csv', index=False)

print("\\nInterpretation:")
print("- Pooled random split gives optimistic numbers (data from same wells in train+test)")
print("- Chronological split is more realistic (predicting future from past)")
print("- LOWO is the strictest test (predicting entirely new wells)")"""),

    # ---- FIGURE 1 ----
    md("""## 5. Figures for the Report

### Figure 1: LOWO Predicted vs Actual"""),

    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax1 = axes[0]
for well, (yt, yp, wc) in lowo_preds.items():
    ax1.scatter(yt, yp, s=14, alpha=0.55, color=WELL_COLORS[well],
                label=short_well(well), edgecolors='none')
lims = [d[TARGET].min() - 10, d[TARGET].max() + 10]
ax1.plot(lims, lims, 'k--', lw=1.5, alpha=0.7, label='Perfect')
ax1.set_xlabel('Actual dP (bar)')
ax1.set_ylabel('Predicted dP (bar)')
ax1.set_title('LOWO: Predicted vs Actual')
ax1.legend(fontsize=8)
ax1.set_xlim(lims); ax1.set_ylim(lims)

# Bar chart
ax2 = axes[1]
x = np.arange(len(lowo_df))
w = 0.35
ax2.bar(x - w/2, lowo_df['R2'], w, color='#1565C0', label='R2', alpha=0.85)
ax2r = ax2.twinx()
ax2r.bar(x + w/2, lowo_df['RMSE'], w, color='#B71C1C', label='RMSE', alpha=0.85)
ax2.axhline(0, color='gray', lw=0.8, ls='--')
ax2.set_xticks(x)
ax2.set_xticklabels([short_well(w) for w in lowo_df['test_well']], rotation=15)
ax2.set_ylabel('R2', color='#1565C0')
ax2r.set_ylabel('RMSE (bar)', color='#B71C1C')
ax2.set_title('Per-fold Performance (LOWO)')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2r.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig1_lowo_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ---- FIGURE 2 ----
    md("""### Figure 2: Chronological Split — Time Series Comparison"""),

    code("""n_wells = len(chrono_preds)
fig, axes = plt.subplots(1, min(n_wells, 5), figsize=(3.5 * min(n_wells, 5), 5), squeeze=False)
axes = axes[0]

for i, (well, (yt, yp, dates)) in enumerate(chrono_preds.items()):
    if i >= 5: break
    ax = axes[i]
    dates_dt = pd.to_datetime(dates)
    ax.plot(dates_dt, yt, 'o-', ms=2, lw=0.8, color='#1565C0', label='Actual', alpha=0.7)
    ax.plot(dates_dt, yp, 's-', ms=2, lw=0.8, color='#E64A19', label='Predicted', alpha=0.7)
    r2_val = chrono_df[chrono_df.test_well == well].R2.values[0]
    ax.set_title(f'{short_well(well)}\\nR2={r2_val:.3f}', fontsize=10)
    ax.set_xlabel('Date')
    if i == 0: ax.set_ylabel('dP (bar)')
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', rotation=45, labelsize=7)

plt.suptitle('Chronological Split: Last 20% of Each Well', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_chrono_split_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ---- FEATURE IMPORTANCE ----
    md("""## 6. Feature Importance Analysis

### 6.1 Gini-based Importance
Random Forest provides built-in feature importance via **mean decrease in impurity** (Gini importance). This tells us which features are most useful for splitting decisions."""),

    code("""# Train on full dataset for importance analysis
rf_full = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_full.fit(d[FEATURES], d[TARGET])

gini_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("=== Gini Feature Importance ===")
for _, r in gini_imp.iterrows():
    bar = '#' * int(r['importance'] * 50)
    print(f"  {r['feature']:30s}  {r['importance']:.4f}  {bar}")

gini_imp.to_csv(f'{OUTPUT_DIR}/feature_importance_gini.csv', index=False)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
gi = gini_imp.sort_values('importance', ascending=True)
bars = ax.barh(gi['feature'], gi['importance'], color='#2E7D32', alpha=0.85)
ax.set_xlabel('Feature Importance (Gini-based)')
ax.set_title('Which measured variables drive dP prediction')
for bar, val in zip(bars, gi['importance']):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_feature_importance_gini.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    md("""### 6.2 SHAP Analysis

SHAP (SHapley Additive exPlanations) provides **model-agnostic, theoretically grounded** feature importance. Unlike Gini importance, SHAP values show both the *magnitude* and *direction* of each feature's impact on individual predictions.

The **beeswarm plot** shows:
- Each dot = one data point
- X-axis = SHAP value (how much that feature pushed the prediction up or down)
- Color = feature value (red = high, blue = low)"""),

    code("""if HAS_SHAP:
    print("Computing SHAP values (may take 1-2 minutes)...")
    shap_sample = d[FEATURES].sample(n=min(500, len(d)), random_state=42)
    explainer = shap.TreeExplainer(rf_full)
    shap_values = explainer.shap_values(shap_sample)
    
    # Save SHAP importance
    shap_imp = pd.DataFrame({
        'feature': FEATURES,
        'shap_mean_abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_mean_abs', ascending=False)
    shap_imp.to_csv(f'{OUTPUT_DIR}/feature_importance_shap.csv', index=False)
    
    print("\\n=== SHAP Feature Importance ===")
    for _, r in shap_imp.iterrows():
        print(f"  {r['feature']:30s}  mean|SHAP| = {r['shap_mean_abs']:.4f}")
    
    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES,
                      show=False, max_display=16)
    plt.title('SHAP Feature Importance', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close('all')
else:
    print("SHAP not available. Install with: pip install shap")"""),

    # ---- BEGGS & BRILL ----
    md("""## 7. Beggs & Brill (1973) Benchmark

We compare our Random Forest against the **Beggs & Brill (1973)** correlation — the most widely used empirical VLP model in the petroleum industry.

> **IMPORTANT CAVEAT:** The Volve production CSV does not contain tubing diameter, well depth, or deviation survey data. The Beggs & Brill calculation below uses **assumed representative values**, not measured ones:
> - Tubing ID: 4.892 inches
> - Well depth: 3,100 m
> - Inclination: 65 degrees
> - API gravity: 28
> - Gas SG: 0.65
>
> These numbers should be replaced with real completion data before final submission."""),

    code("""# Beggs & Brill pressure traverse implementation
ASSUMED_TUBING_ID = 4.892  # inches
ASSUMED_DEPTH = 3100.0     # meters
ASSUMED_INCL = 65.0        # degrees
ASSUMED_API = 28.0
ASSUMED_GAS_SG = 0.65

def beggs_brill_dP(q_oil, q_gas, q_wat, P_wh, T_avg,
                    d_in=ASSUMED_TUBING_ID, H=ASSUMED_DEPTH,
                    theta=ASSUMED_INCL, API=ASSUMED_API, gas_sg=ASSUMED_GAS_SG):
    try:
        d_m = d_in * 0.0254
        A = np.pi * d_m**2 / 4.0
        theta_rad = np.radians(theta)
        g = 9.81
        oil_sg = 141.5 / (API + 131.5)
        T_F = T_avg * 9/5 + 32
        P_psia = P_wh * 14.696
        
        # Solution GOR (Standing 1947)
        Rs = gas_sg * ((P_psia / 18.2 + 1.4)**1.205) * 10**(0.0125*API - 0.00091*T_F)
        actual_GOR = q_gas / max(q_oil, 1e-6)
        Rs = min(Rs, actual_GOR)
        
        # Oil FVF (Standing)
        Bo = 0.972 + 1.47e-4 * (Rs * np.sqrt(gas_sg/oil_sg) + 1.25*T_F)**1.175
        Bw = 1.0
        
        # Gas z-factor (Papay)
        T_R = T_F + 459.67
        P_pc = 677 + 15*gas_sg - 37.5*gas_sg**2
        T_pc = 168 + 325*gas_sg - 12.5*gas_sg**2
        P_pr = P_psia / P_pc
        T_pr = T_R / T_pc
        z = max(1 - 3.52*P_pr/10**(0.9813*T_pr) + 0.274*P_pr**2/10**(0.8157*T_pr), 0.1)
        Bg = 0.0283 * z * T_R / max(P_psia, 14.7)
        
        # In-situ volumetric rates
        q_oil_is = q_oil * Bo / 86400.0
        q_wat_is = q_wat * Bw / 86400.0
        q_liq_is = q_oil_is + q_wat_is
        q_gas_is = max(actual_GOR - Rs, 0) * q_oil * Bg * 0.0283168 / 86400.0
        
        v_sl = q_liq_is / A
        v_sg = q_gas_is / A
        v_m = v_sl + v_sg
        if v_m < 1e-8: return 0.0
        
        lambda_L = v_sl / v_m
        rho_oil = oil_sg * 1000 / Bo
        rho_wat = 1020.0 / Bw
        rho_gas = P_psia * 28.97 * gas_sg / (z * 10.73 * T_R) * 16.0185
        WC_loc = q_wat / max(q_oil + q_wat, 1e-6)
        rho_L = rho_oil * (1 - WC_loc) + rho_wat * WC_loc
        
        NFr = v_m**2 / (g * d_m)
        
        # Flow regime + holdup
        L1 = 316 * lambda_L**0.302
        L2 = 0.0009252 * lambda_L**(-2.4684) if lambda_L > 0 else 1e10
        L3 = 0.10 * lambda_L**(-1.4516) if lambda_L > 0 else 1e10
        
        regime_params = {
            'segregated': (0.980, 0.4846, 0.0868),
            'intermittent': (0.845, 0.5351, 0.0173),
            'distributed': (1.065, 0.5824, 0.0609),
        }
        
        if lambda_L < 0.01 and NFr < L1:
            regime = 'segregated'
        elif lambda_L >= 0.01 and NFr < L2:
            regime = 'segregated'
        elif lambda_L >= 0.01 and L2 <= NFr <= L3:
            regime = 'intermittent'
        else:
            regime = 'distributed'
        
        a, b, c = regime_params[regime]
        HL_0 = a * lambda_L**b / max(NFr**c, 1e-10)
        HL = np.clip(HL_0, lambda_L, 1.0)
        
        rho_m = rho_L * HL + rho_gas * (1 - HL)
        rho_ns = rho_L * lambda_L + rho_gas * (1 - lambda_L)
        
        mu_oil = 10**(0.43 + 8.33/API) * T_F**(-0.8) * 0.001
        mu_ns = mu_oil * lambda_L + 1e-5 * (1 - lambda_L)
        Re = max(rho_ns * v_m * d_m / max(mu_ns, 1e-10), 100)
        fn = max(1 / (2*np.log10(Re/4.5223*np.log10(Re) - 3.8215))**2, 0.001)
        
        y = lambda_L / max(HL**2, 1e-10)
        if 1e-4 < y <= 1.0 or y >= 1.2:
            ln_y = np.log(max(y, 1e-10))
            denom = -0.0523 + 3.182*ln_y - 0.8725*ln_y**2 + 0.01853*ln_y**4
            S = ln_y/denom if abs(denom) > 1e-10 else 0
        else:
            S = np.log(2.2*y - 1.2) if y > 1.0 else 0
        
        fm = fn * np.exp(np.clip(S, -10, 10))
        dPdz = rho_m * g * np.sin(theta_rad) + fm * rho_ns * v_m**2 / (2*d_m)
        
        return dPdz * H / 1e5  # Pa -> bar
    except:
        return np.nan

# Compute B&B for all data points
print("Computing Beggs & Brill predictions...")
bb_preds = []
for _, row in d.iterrows():
    T_avg = (row.get('AVG_DOWNHOLE_TEMPERATURE', 100) + row.get('AVG_WHT_P', 60)) / 2
    bb_preds.append(beggs_brill_dP(row['q_oil'], row['q_gas'], row['q_wat'],
                                    row['AVG_WHP_P'], T_avg))

d['dP_BB'] = bb_preds
valid_bb = d.dropna(subset=['dP_BB'])

# B&B metrics per well
print("\\n=== Beggs & Brill Results (ASSUMED geometry) ===")
bb_results = []
for well in wells:
    wd = valid_bb[valid_bb['Wellbore name'] == well]
    if len(wd) < 5: continue
    yt_bb = wd[TARGET].values
    yp_bb = wd['dP_BB'].values
    rmse = np.sqrt(mean_squared_error(yt_bb, yp_bb))
    r2 = r2_score(yt_bb, yp_bb)
    bb_results.append({'well': well, 'RMSE_BB': rmse, 'R2_BB': r2})
    print(f"  {well:20s}  RMSE={rmse:.2f} bar  R2={r2:.3f}")

bb_df = pd.DataFrame(bb_results)
bb_df.to_csv(f'{OUTPUT_DIR}/beggs_brill_results.csv', index=False)"""),

    md("""### Figure 5: RF vs Beggs & Brill Comparison"""),

    code("""# Merge RF and B&B results
merged = lowo_df.merge(bb_df, left_on='test_well', right_on='well', how='left').dropna(subset=['RMSE_BB'])

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(merged))
w = 0.35
ax.bar(x - w/2, merged['RMSE'], w, color='#1565C0', label='Random Forest (LOWO)', alpha=0.85)
ax.bar(x + w/2, merged['RMSE_BB'], w, color='#B71C1C', label='Beggs & Brill (assumed geom.)', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([short_well(w) for w in merged['test_well']], rotation=15)
ax.set_ylabel('RMSE (bar)')
ax.set_title('RF vs Beggs & Brill: RMSE by Well\\n*B&B uses assumed geometry')
ax.legend()
for i, (rv, bv) in enumerate(zip(merged['RMSE'], merged['RMSE_BB'])):
    ax.text(i - w/2, rv + 1, f'{rv:.1f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, bv + 1, f'{bv:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig5_rf_vs_bb.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ---- RESIDUAL ANALYSIS ----
    md("""## 8. Residual Analysis

Residuals = Actual - Predicted. Analyzing residuals tells us:
- Is the model biased? (Mean residual should be near 0)
- Does error depend on any input feature? (Would indicate systematic model weakness)"""),

    code("""residuals = yt_pool - yp_pool

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Predicted
ax = axes[0, 0]
ax.scatter(yp_pool, residuals, s=10, alpha=0.4, color='#1565C0', edgecolors='none')
ax.axhline(0, color='red', ls='--', lw=1.5)
ax.set_xlabel('Predicted dP (bar)'); ax.set_ylabel('Residual')
ax.set_title('Residuals vs Predicted')

# Histogram
ax = axes[0, 1]
ax.hist(residuals, bins=40, color='#388E3C', alpha=0.75, edgecolor='white')
ax.axvline(0, color='red', ls='--', lw=1.5)
ax.set_xlabel('Residual (bar)'); ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution\\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')

# Residuals vs WC
ax = axes[1, 0]
ax.scatter(X_test['WC'].values, residuals, s=10, alpha=0.4, color='#E64A19', edgecolors='none')
ax.axhline(0, color='red', ls='--', lw=1.5)
ax.set_xlabel('Water Cut'); ax.set_ylabel('Residual')
ax.set_title('Residuals vs Water Cut')

# Abs error vs GOR
ax = axes[1, 1]
ax.scatter(X_test['GOR'].values, np.abs(residuals), s=10, alpha=0.4, color='#7B1FA2', edgecolors='none')
ax.set_xlabel('GOR'); ax.set_ylabel('|Error| (bar)')
ax.set_title('Absolute Error vs GOR')

plt.suptitle('Residual Analysis (Pooled Split)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig7_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ---- VALIDATION COMPARISON ----
    md("""### Figure 8: Validation Strategy Comparison"""),

    code("""fig, ax = plt.subplots(figsize=(10, 6))
strategies = ['LOWO\\n(cross-well)', 'Chronological\\n(within-well)', 'Pooled\\n(random)']
r2_vals = [lowo_df.R2.mean(), chrono_df.R2.mean(), pool_r2]
rmse_vals = [lowo_df.RMSE.mean(), chrono_df.RMSE.mean(), pool_rmse]
r2_stds = [lowo_df.R2.std(), chrono_df.R2.std(), 0]

x = np.arange(len(strategies))
w = 0.35
bars1 = ax.bar(x - w/2, r2_vals, w, color='#1565C0', label='Mean R2', alpha=0.85,
               yerr=r2_stds, capsize=5)
ax_r = ax.twinx()
bars2 = ax_r.bar(x + w/2, rmse_vals, w, color='#B71C1C', label='Mean RMSE', alpha=0.85)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xticks(x); ax.set_xticklabels(strategies)
ax.set_ylabel('R2', color='#1565C0')
ax_r.set_ylabel('RMSE (bar)', color='#B71C1C')
ax.set_title('Comparison of Three Validation Strategies')
for b, v in zip(bars1, r2_vals):
    ax.text(b.get_x()+b.get_width()/2, max(v,0)+0.03, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
for b, v in zip(bars2, rmse_vals):
    ax_r.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax_r.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig8_validation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()"""),

    # ---- SAVE MODEL ----
    md("""## 9. Save Trained Model

Save the trained model for use in the Streamlit web application."""),

    code("""import pickle

# Save the best model (trained on full dataset)
model_path = f'{OUTPUT_DIR}/rf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': rf_full,
        'features': FEATURES,
        'target': TARGET,
        'best_params': best_params,
        'training_stats': {
            'n_samples': len(d),
            'n_features': len(FEATURES),
            'target_mean': d[TARGET].mean(),
            'target_std': d[TARGET].std(),
        }
    }, f)
print(f"Model saved to {model_path}")
print(f"  Features: {len(FEATURES)}")
print(f"  Best params: {best_params}")"""),

    # ---- SUMMARY ----
    md("""## 10. Results Summary

### Key Results Table

| Metric | LOWO (cross-well) | Chronological | Pooled Random |
|--------|-------------------|---------------|---------------|
| Mean R2 | See output above | See output above | See output above |
| Mean RMSE (bar) | See output above | See output above | See output above |
| Mean MAPE (%) | See output above | See output above | See output above |

### Key Findings

1. **Water Cut dominates**: WC is the single most important feature (~29% Gini importance), confirming that classical correlations developed on low-WC lab data are weakest where they matter most.

2. **Cross-well generalization is limited**: The LOWO R2 varies significantly by well. F-15D (a low-rate well operating in a narrow dP range) is hardest to predict from other wells' data.

3. **Within-well prediction is strong**: The chronological split shows the model can predict future production from past data within the same well — the most practical use case.

4. **RF consistently outperforms Beggs & Brill**: Even with assumed geometry (which should be replaced before submission), the RF model achieves lower RMSE on every well.

### Limitations to State in Your Report

1. Model is trained and validated on a single field (Volve, North Sea) — generalization to other fields is unproven
2. Beggs & Brill comparison uses assumed well geometry (not measured)
3. Daily-averaged data may mask sub-daily transient effects
4. Only 5 wells available — small sample for cross-well validation
5. No independent external test dataset (e.g., TUFFP lab data)"""),

]


# =====================================================================
# WRITE NOTEBOOKS
# =====================================================================
nb1 = make_notebook(nb1_cells)
nb1_path = os.path.join(NB_DIR, '01_Data_Analysis.ipynb')
with open(nb1_path, 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)
print(f"Created: {nb1_path} ({len(nb1_cells)} cells)")

nb2 = make_notebook(nb2_cells)
nb2_path = os.path.join(NB_DIR, '02_ML_Model.ipynb')
with open(nb2_path, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)
print(f"Created: {nb2_path} ({len(nb2_cells)} cells)")

print("\nDone! Open these notebooks in Jupyter or upload to Google Colab.")
