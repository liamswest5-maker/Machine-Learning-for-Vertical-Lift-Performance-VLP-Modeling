"""
=============================================================================
VLP RANDOM FOREST MODEL v2.0 - COMPREHENSIVE PIPELINE
Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow
=============================================================================

This is the COMPLETE, FINAL pipeline for the project. It produces:
  - Cleaned, feature-engineered dataset
  - Leave-One-Well-Out (LOWO) cross-validation results
  - Chronological within-well train/test split results
  - Pooled random split results (for comparison)
  - SHAP feature importance analysis
  - Beggs & Brill benchmark (with assumed geometry - caveat documented)
  - All figures for the final report (8 publication-quality figures)
  - All metrics CSVs for the results chapter

INPUT:  data/volve_welldata_raw.csv
OUTPUT: data/*.csv, figures/*.png
"""

import pandas as pd
import numpy as np
import warnings
import sys, io, os

# Handle Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Try to import SHAP - install if not available
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARNING] SHAP not installed. Run: pip install shap")
    print("         SHAP analysis will be skipped.\n")

# ====================== CONFIGURATION ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'volve_welldata_raw.csv')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

USABLE_WELLS = ['15/9-F-1 C', '15/9-F-11 H', '15/9-F-12 H',
                '15/9-F-14 H', '15/9-F-15 D']

# Assumed well geometry (Volve completion reports not in raw CSV)
# These are representative North Sea values - flagged as assumptions
ASSUMED_TUBING_ID_INCHES = 4.892  # ~6-5/8" casing with 4.892" tubing
ASSUMED_WELL_DEPTH_M = 3100.0     # Approximate Volve reservoir TVD
ASSUMED_INCLINATION_DEG = 65.0    # Deviated wells
ASSUMED_API = 28.0                # Medium oil
ASSUMED_GAS_SG = 0.65             # Gas specific gravity

# Plot styling
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

WELL_COLORS = {
    '15/9-F-1 C':  '#1976D2',
    '15/9-F-11 H': '#388E3C',
    '15/9-F-12 H': '#E64A19',
    '15/9-F-14 H': '#7B1FA2',
    '15/9-F-15 D': '#00838F',
}

def short_well(name):
    return name.replace('15/9-', '')


# =============================================================================
# STAGE 1 - LOAD AND FILTER
# =============================================================================
print("=" * 72)
print("  STAGE 1/8: Load and filter raw Volve production data")
print("=" * 72)

df = pd.read_csv(INPUT_PATH)
print(f"  Raw file: {len(df)} rows, {df.columns.size} columns")

# Keep only producer rows for usable wells
op = df[df['WELL_TYPE'] == 'OP'].copy()
op = op[op['Wellbore name'].isin(USABLE_WELLS)].copy()
print(f"  Producer rows for usable wells: {len(op)}")

# Coerce numeric columns
NUM_COLS = ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
            'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
for c in NUM_COLS:
    op[c] = pd.to_numeric(op[c], errors='coerce')

# Core validity filter: must have PDG data, positive rates, valid tubing dP
mask = ((op['AVG_DOWNHOLE_PRESSURE'] > 0) & (op['AVG_WHP_P'] > 0) &
        (op['ON_STREAM_HRS'] > 0) & (op['BORE_OIL_VOL'] > 0) &
        (op['AVG_DP_TUBING'].notna()) & (op['AVG_DP_TUBING'] > 0))
clean = op[mask].copy()
print(f"  After validity filter: {len(clean)} rows")


# =============================================================================
# STAGE 2 - FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 2/8: Feature engineering")
print("=" * 72)

# Parse dates properly (mixed format in raw file)
clean['Date of Production'] = pd.to_datetime(clean['Date of Production'], format='mixed')
clean = clean.sort_values(['Wellbore name', 'Date of Production']).reset_index(drop=True)

# Normalize rates by on-stream hours (correct for partial-day production)
clean['q_oil'] = clean['BORE_OIL_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_gas'] = clean['BORE_GAS_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_wat'] = clean['BORE_WAT_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_liq'] = clean['q_oil'] + clean['q_wat']

# Compositional features
clean['WC']  = clean['q_wat'] / clean['q_liq'].replace(0, np.nan)
clean['GOR'] = clean['q_gas'] / clean['q_oil'].replace(0, np.nan)

# Log-transformed rates (better for tree models with wide ranges)
clean['log_q_liq'] = np.log1p(clean['q_liq'])
clean['log_q_oil'] = np.log1p(clean['q_oil'])
clean['log_q_gas'] = np.log1p(clean['q_gas'])

# Gas-liquid ratio (alternative to GOR for wells producing water)
clean['GLR'] = clean['q_gas'] / clean['q_liq'].replace(0, np.nan)

# Pressure ratio (dimensionless - captures compression ratio)
clean['P_ratio'] = clean['AVG_DOWNHOLE_PRESSURE'] / clean['AVG_WHP_P'].replace(0, np.nan)

# Temperature gradient proxy (dT between downhole and wellhead)
clean['dT'] = clean['AVG_DOWNHOLE_TEMPERATURE'] - clean['AVG_WHT_P']

# Target variable: wellbore pressure drop
clean['delta_P'] = clean['AVG_DOWNHOLE_PRESSURE'] - clean['AVG_WHP_P']

# Physical consistency filters
clean = clean[(clean['WC'] >= 0) & (clean['WC'] <= 1) &
              (clean['GOR'] > 0) & (clean['GOR'] < 5000) &
              (clean['q_liq'] > 0) & (clean['delta_P'] > 0)]

print(f"  After physical consistency filter: {len(clean)} rows")
print(f"  Features added: q_oil, q_gas, q_wat, q_liq, WC, GOR, GLR, log rates,")
print(f"                  P_ratio, dT, delta_P (target)")


# =============================================================================
# STAGE 3 - STEADY-STATE FILTER
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 3/8: Steady-state filter (7-day rolling window)")
print("=" * 72)

g = clean.groupby('Wellbore name', group_keys=False)
clean['roll_std_q']  = g['q_liq'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['roll_mean_q'] = g['q_liq'].rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
clean['roll_std_dP'] = g['delta_P'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['cv_q'] = clean['roll_std_q'] / clean['roll_mean_q'].replace(0, np.nan)

# Thresholds chosen from actual data distribution (see diagnose_data.py)
CV_THRESHOLD, DP_STD_THRESHOLD = 0.35, 30.0
steady_mask = ((clean['cv_q'] < CV_THRESHOLD) &
               (clean['roll_std_dP'] < DP_STD_THRESHOLD) &
               clean['cv_q'].notna())
steady = clean[steady_mask].copy()

print(f"  Retained: {len(steady)} of {len(clean)} rows ({len(steady)/len(clean)*100:.1f}%)")
print(f"\n  Per-well breakdown:")
for well in USABLE_WELLS:
    n = len(steady[steady['Wellbore name'] == well])
    print(f"    {short_well(well):8s}: {n:5d} rows")


# =============================================================================
# DEFINE FEATURE SETS
# =============================================================================

# Core features: directly measured or simply derived from measured columns
FEATURES_CORE = ['q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
                 'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
                 'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS']

# Extended features: adds engineered features for better model performance
FEATURES_EXTENDED = FEATURES_CORE + ['log_q_liq', 'log_q_oil', 'log_q_gas',
                                      'GLR', 'dT']

TARGET = 'delta_P'

# Use extended features for the main model
FEATURES = FEATURES_EXTENDED
d = steady[['Wellbore name', 'Date of Production'] + FEATURES + [TARGET]].dropna().copy()
print(f"\n  Final model-ready dataset: {len(d)} rows x {len(FEATURES)} features")


# =============================================================================
# STAGE 4 - HYPERPARAMETER TUNING (GridSearchCV)
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 4/8: Hyperparameter tuning via GridSearchCV")
print("=" * 72)

# Use 80% of data for tuning, stratified by well
X_tune, _, y_tune, _ = train_test_split(
    d[FEATURES], d[TARGET], test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [300, 500],
    'min_samples_leaf': [1, 3],
    'max_features': ['sqrt', 0.5],
}

n_combos = np.prod([len(v) for v in param_grid.values()])
print(f"  Grid search over {n_combos} "
      f"parameter combinations (3-fold CV)...")

gs = GridSearchCV(
    RandomForestRegressor(max_depth=None, random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring='r2', n_jobs=1, verbose=1)
gs.fit(X_tune, y_tune)

best_params = gs.best_params_
best_params['max_depth'] = None  # Keep unbounded depth
print(f"  Best parameters: {best_params}")
print(f"  Best CV R2: {gs.best_score_:.4f}")


# =============================================================================
# STAGE 5 - MODEL VALIDATION (Three strategies)
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 5/8: Model validation (3 strategies)")
print("=" * 72)

wells = d['Wellbore name'].unique()

# --- Strategy A: Leave-One-Well-Out (LOWO) ---
print("\n  --- A. Leave-One-Well-Out Cross-Validation ---")
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
print(f"\n  {'Well':20s}  {'n_test':>6s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>7s}  {'R2':>8s}")
for _, r in lowo_df.iterrows():
    print(f"  {r['test_well']:20s}  {int(r['n_test']):6d}  {r['RMSE']:8.2f}  "
          f"{r['MAE']:8.2f}  {r['MAPE']:7.2f}  {r['R2']:8.3f}")
print(f"\n  Mean:  RMSE={lowo_df.RMSE.mean():.2f}  MAPE={lowo_df.MAPE.mean():.2f}%  R2={lowo_df.R2.mean():.3f}")
print(f"  Std:   RMSE={lowo_df.RMSE.std():.2f}  MAPE={lowo_df.MAPE.std():.2f}%  R2={lowo_df.R2.std():.3f}")

lowo_df.to_csv(os.path.join(DATA_DIR, 'lowo_results.csv'), index=False)

# --- Strategy B: Chronological Within-Well Split ---
print("\n  --- B. Chronological Within-Well Split (80/20) ---")
chrono_results = []
chrono_preds = {}

for well in wells:
    well_data = d[d['Wellbore name'] == well].sort_values('Date of Production')
    split_idx = int(len(well_data) * 0.8)
    train_df = well_data.iloc[:split_idx]
    test_df  = well_data.iloc[split_idx:]

    if len(test_df) < 5:
        continue

    # Train on ALL other wells + early data from this well
    other_wells = d[d['Wellbore name'] != well]
    train_combined = pd.concat([other_wells, train_df])

    rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf.fit(train_combined[FEATURES], train_combined[TARGET])

    yp = rf.predict(test_df[FEATURES])
    yt = test_df[TARGET].values

    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    r2 = r2_score(yt, yp)

    chrono_results.append({
        'test_well': well, 'n_test': len(test_df), 'n_train': len(train_combined),
        'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
    })
    chrono_preds[well] = (yt, yp, test_df['Date of Production'].values)

chrono_df = pd.DataFrame(chrono_results)
print(f"\n  {'Well':20s}  {'n_test':>6s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>7s}  {'R2':>8s}")
for _, r in chrono_df.iterrows():
    print(f"  {r['test_well']:20s}  {int(r['n_test']):6d}  {r['RMSE']:8.2f}  "
          f"{r['MAE']:8.2f}  {r['MAPE']:7.2f}  {r['R2']:8.3f}")
print(f"\n  Mean:  RMSE={chrono_df.RMSE.mean():.2f}  MAPE={chrono_df.MAPE.mean():.2f}%  R2={chrono_df.R2.mean():.3f}")

chrono_df.to_csv(os.path.join(DATA_DIR, 'chrono_split_results.csv'), index=False)

# --- Strategy C: Pooled Random Split ---
print("\n  --- C. Pooled Random Split (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    d[FEATURES], d[TARGET], test_size=0.2, random_state=42)

rf_pooled = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_pooled.fit(X_train, y_train)
yp_pool = rf_pooled.predict(X_test)
yt_pool = y_test.values

pool_rmse = np.sqrt(mean_squared_error(yt_pool, yp_pool))
pool_mae = mean_absolute_error(yt_pool, yp_pool)
pool_mape = np.mean(np.abs((yt_pool - yp_pool) / yt_pool)) * 100
pool_r2 = r2_score(yt_pool, yp_pool)

print(f"  Train: {len(X_train)},  Test: {len(X_test)}")
print(f"  RMSE={pool_rmse:.2f}  MAE={pool_mae:.2f}  MAPE={pool_mape:.2f}%  R2={pool_r2:.4f}")

pooled_df = pd.DataFrame([{
    'split': 'Pooled 80/20', 'n_train': len(X_train), 'n_test': len(X_test),
    'RMSE': pool_rmse, 'MAE': pool_mae, 'MAPE': pool_mape, 'R2': pool_r2
}])
pooled_df.to_csv(os.path.join(DATA_DIR, 'pooled_split_results.csv'), index=False)


# =============================================================================
# STAGE 5.5 - CLASSIFICATION-STYLE METRICS (Accuracy, Precision, Recall, F1)
# =============================================================================
# Rationale: The model predicts a continuous ΔP (regression). However, for
# reservoir/production engineering decisions, what matters is whether the model
# correctly identifies the FLOW REGIME (Low / Medium / High pressure drop).
#
# We convert predictions to regime classes and compute real classification
# metrics (sklearn precision_score, recall_score, f1_score, accuracy_score).
#
# Regime thresholds are based on the data distribution:
#   Low    ΔP < 100 bar  → nearing depletion, low energy
#   Medium 100 ≤ ΔP < 200 bar → normal production
#   High   ΔP ≥ 200 bar  → high-energy / high-rate conditions
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 5.5/8: Classification-style metrics (regime classification)")
print("=" * 72)
print("  Regime bins:  Low (<100 bar)  |  Medium (100–200 bar)  |  High (>200 bar)")

from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

DP_BINS   = [0, 100, 200, np.inf]
DP_LABELS = ['Low', 'Medium', 'High']

def classify_regime(y_array):
    """Bin continuous ΔP values into Low / Medium / High regime labels."""
    return pd.cut(y_array, bins=DP_BINS, labels=DP_LABELS, right=False).astype(str)

def compute_clf_metrics(y_true_arr, y_pred_arr, strategy_name, fold_name=None):
    """
    Given arrays of actual and predicted ΔP (continuous), bin both into
    regime classes and compute precision, recall, F1, and accuracy.
    Returns a list of per-class metric dicts.
    """
    y_true_cls = classify_regime(y_true_arr)
    y_pred_cls = classify_regime(y_pred_arr)

    # Use macro average for the overall scores
    acc  = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, average='macro',
                           labels=DP_LABELS, zero_division=0)
    rec  = recall_score(y_true_cls, y_pred_cls, average='macro',
                        labels=DP_LABELS, zero_division=0)
    f1   = f1_score(y_true_cls, y_pred_cls, average='macro',
                    labels=DP_LABELS, zero_division=0)

    # Per-class breakdown
    rows = []
    per_prec = precision_score(y_true_cls, y_pred_cls, average=None,
                               labels=DP_LABELS, zero_division=0)
    per_rec  = recall_score(y_true_cls, y_pred_cls, average=None,
                            labels=DP_LABELS, zero_division=0)
    per_f1   = f1_score(y_true_cls, y_pred_cls, average=None,
                        labels=DP_LABELS, zero_division=0)
    per_sup  = [(y_true_cls == lbl).sum() for lbl in DP_LABELS]

    for i, lbl in enumerate(DP_LABELS):
        rows.append({
            'Strategy': strategy_name,
            'Fold': fold_name if fold_name else 'All',
            'Regime': lbl,
            'Precision': round(per_prec[i], 4),
            'Recall':    round(per_rec[i],  4),
            'F1':        round(per_f1[i],   4),
            'Support':   int(per_sup[i]),
        })

    # Summary row (macro averages)
    rows.append({
        'Strategy': strategy_name,
        'Fold': fold_name if fold_name else 'All',
        'Regime': 'MACRO AVG',
        'Precision': round(prec, 4),
        'Recall':    round(rec,  4),
        'F1':        round(f1,   4),
        'Support':   int(len(y_true_arr)),
    })

    return rows, acc, prec, rec, f1


all_clf_rows = []

# ── A. LOWO ──────────────────────────────────────────────────────────────────
print("\n  A. LOWO classification metrics:")
for well, (yt, yp, _) in lowo_preds.items():
    rows, acc, prec, rec, f1 = compute_clf_metrics(yt, yp, 'LOWO', short_well(well))
    all_clf_rows.extend(rows)
    print(f"    {short_well(well):8s}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}")

# ── B. Chronological ─────────────────────────────────────────────────────────
print("\n  B. Chrono classification metrics:")
for well, (yt, yp, _) in chrono_preds.items():
    rows, acc, prec, rec, f1 = compute_clf_metrics(yt, yp, 'Chrono', short_well(well))
    all_clf_rows.extend(rows)
    print(f"    {short_well(well):8s}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}")

# ── C. Pooled ────────────────────────────────────────────────────────────────
print("\n  C. Pooled classification metrics:")
rows, acc, prec, rec, f1 = compute_clf_metrics(yt_pool, yp_pool, 'Pooled')
all_clf_rows.extend(rows)
print(f"    All    Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

clf_df = pd.DataFrame(all_clf_rows)
clf_df.to_csv(os.path.join(DATA_DIR, 'classification_metrics.csv'), index=False)
print(f"\n  Saved: data/classification_metrics.csv ({len(clf_df)} rows)")

# Per-strategy macro-average summary for easy reading
print("\n  ── Macro-Average Summary (regime classification) ──")
macro = clf_df[clf_df['Regime'] == 'MACRO AVG'].groupby('Strategy')[
    ['Precision', 'Recall', 'F1']].mean().round(4)
print(macro.to_string())


# =============================================================================
# STAGE 6 - FEATURE IMPORTANCE (Gini + SHAP)
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 6/8: Feature importance analysis")
print("=" * 72)

# Train on full dataset for importance analysis
rf_full = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
rf_full.fit(d[FEATURES], d[TARGET])

# Gini importance
gini_imp = pd.DataFrame({
    'feature': FEATURES,
    'gini_importance': rf_full.feature_importances_
}).sort_values('gini_importance', ascending=False)
gini_imp.to_csv(os.path.join(DATA_DIR, 'feature_importance_gini.csv'), index=False)

print("\n  Gini-based feature importance:")
for _, r in gini_imp.iterrows():
    bar = '#' * int(r['gini_importance'] * 50)
    print(f"    {r['feature']:30s}  {r['gini_importance']:.4f}  {bar}")

# SHAP importance
if HAS_SHAP:
    print("\n  Computing SHAP values (this may take 1-2 minutes)...")
    # Use a subsample for speed
    shap_sample = d[FEATURES].sample(n=min(200, len(d)), random_state=42)
    explainer = shap.TreeExplainer(rf_full)
    shap_values = explainer.shap_values(shap_sample)

    shap_imp = pd.DataFrame({
        'feature': FEATURES,
        'shap_mean_abs': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_mean_abs', ascending=False)
    shap_imp.to_csv(os.path.join(DATA_DIR, 'feature_importance_shap.csv'), index=False)

    print("\n  SHAP-based feature importance (mean |SHAP|):")
    for _, r in shap_imp.iterrows():
        bar = '#' * int(r['shap_mean_abs'] / shap_imp['shap_mean_abs'].max() * 30)
        print(f"    {r['feature']:30s}  {r['shap_mean_abs']:.4f}  {bar}")
else:
    shap_values = None
    shap_sample = None


# =============================================================================
# STAGE 7 - BEGGS & BRILL BENCHMARK
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 7/8: Beggs & Brill benchmark (ASSUMED geometry)")
print("=" * 72)
print("  WARNING: Uses assumed tubing geometry, not measured values.")
print(f"  Assumed: ID={ASSUMED_TUBING_ID_INCHES}in, Depth={ASSUMED_WELL_DEPTH_M}m, "
      f"Inc={ASSUMED_INCLINATION_DEG}deg")

def beggs_brill_dP(q_oil, q_gas, q_wat, P_wh, T_avg,
                    d_in=ASSUMED_TUBING_ID_INCHES,
                    H=ASSUMED_WELL_DEPTH_M,
                    theta=ASSUMED_INCLINATION_DEG,
                    API=ASSUMED_API, gas_sg=ASSUMED_GAS_SG):
    """
    Beggs & Brill (1973) pressure traverse calculation.
    Returns predicted delta_P (bar) for given conditions.
    """
    try:
        d_m = d_in * 0.0254  # convert to meters
        A = np.pi * d_m**2 / 4.0
        theta_rad = np.radians(theta)
        g = 9.81

        # PVT: Standing correlations
        oil_sg = 141.5 / (API + 131.5)
        T_F = T_avg * 9/5 + 32  # C to F
        P_psia = P_wh * 14.696  # bar to psia (use average pressure)

        # Solution GOR (Standing 1947) - limited to actual producing GOR
        Rs = gas_sg * ((P_psia / 18.2 + 1.4) ** 1.205) * 10**(0.0125 * API - 0.00091 * T_F)
        actual_GOR = q_gas / max(q_oil, 1e-6)
        Rs = min(Rs, actual_GOR)

        # Oil FVF
        Bo = 0.972 + 1.47e-4 * (Rs * np.sqrt(gas_sg / oil_sg) + 1.25 * T_F) ** 1.175

        # Water FVF (approximately 1 at these conditions)
        Bw = 1.0

        # Gas z-factor (Papay correlation)
        T_R = (T_F + 459.67)  # Rankine
        P_pc = 677 + 15 * gas_sg - 37.5 * gas_sg**2
        T_pc = 168 + 325 * gas_sg - 12.5 * gas_sg**2
        P_pr = P_psia / P_pc
        T_pr = T_R / T_pc
        z = 1 - 3.52 * P_pr / 10**(0.9813 * T_pr) + 0.274 * P_pr**2 / 10**(0.8157 * T_pr)
        z = max(z, 0.1)

        # Gas FVF
        Bg = 0.0283 * z * T_R / max(P_psia, 14.7)  # ft3/scf

        # In-situ volumetric rates (m3/s)
        # Input rates are Sm3/d, convert to m3/s at downhole conditions
        q_oil_is = q_oil * Bo / 86400.0
        q_wat_is = q_wat * Bw / 86400.0
        q_liq_is = q_oil_is + q_wat_is
        q_gas_is = max(actual_GOR - Rs, 0) * q_oil * Bg * 0.0283168 / 86400.0  # ft3->m3

        # Superficial velocities
        v_sl = q_liq_is / A
        v_sg = q_gas_is / A
        v_m = v_sl + v_sg

        if v_m < 1e-8:
            return 0.0

        # No-slip holdup
        lambda_L = v_sl / v_m

        # Fluid properties at average conditions
        rho_oil = oil_sg * 1000 / Bo
        rho_wat = 1020.0 / Bw
        rho_gas = P_psia * 28.97 * gas_sg / (z * 10.73 * T_R) * 16.0185  # kg/m3

        WC_local = q_wat / max(q_oil + q_wat, 1e-6)
        rho_L = rho_oil * (1 - WC_local) + rho_wat * WC_local

        # Froude number for flow regime determination
        Fr = v_m**2 / (g * d_m)
        NFr = Fr

        # Flow regime boundaries (Beggs & Brill 1973)
        L1 = 316 * lambda_L**0.302
        L2 = 0.0009252 * lambda_L**(-2.4684)
        L3 = 0.10 * lambda_L**(-1.4516)
        L4 = 0.5 * lambda_L**(-6.738)

        # Determine flow regime
        if (lambda_L < 0.01 and NFr < L1) or (lambda_L >= 0.01 and NFr < L2):
            regime = 'segregated'
        elif (lambda_L >= 0.01 and L2 <= NFr <= L3):
            regime = 'transition'
        elif ((0.01 <= lambda_L < 0.4 and L3 < NFr <= L1) or
              (lambda_L >= 0.4 and L3 < NFr <= L4)):
            regime = 'intermittent'
        else:
            regime = 'distributed'

        # Horizontal holdup
        regime_params = {
            'segregated':   (0.980, 0.4846, 0.0868),
            'intermittent': (0.845, 0.5351, 0.0173),
            'distributed':  (1.065, 0.5824, 0.0609),
        }

        if regime == 'transition':
            # Interpolate between segregated and intermittent
            if L3 > L2:
                A_frac = (L3 - NFr) / (L3 - L2)
            else:
                A_frac = 0.5
            A_frac = np.clip(A_frac, 0, 1)
            a_s, b_s, c_s = regime_params['segregated']
            a_i, b_i, c_i = regime_params['intermittent']
            HL_seg = a_s * lambda_L**b_s / max(NFr**c_s, 1e-10)
            HL_int = a_i * lambda_L**b_i / max(NFr**c_i, 1e-10)
            HL_0 = A_frac * HL_seg + (1 - A_frac) * HL_int
        else:
            a, b, c = regime_params[regime]
            HL_0 = a * lambda_L**b / max(NFr**c, 1e-10)

        HL_0 = np.clip(HL_0, lambda_L, 1.0)

        # Inclination correction
        incl_params = {
            'segregated':   (0.011, -3.768, 3.539, -1.614),
            'intermittent': (2.960, 0.305, -0.4473, 0.0978),
            'distributed':  (0, 0, 0, 0),
        }

        if regime == 'transition':
            C_seg = _bb_C(lambda_L, NFr, incl_params['segregated'])
            C_int = _bb_C(lambda_L, NFr, incl_params['intermittent'])
            C = A_frac * C_seg + (1 - A_frac) * C_int
        elif regime in incl_params:
            C = _bb_C(lambda_L, NFr, incl_params[regime])
        else:
            C = 0

        psi = 1 + C * (np.sin(1.8 * theta_rad) - (1/3) * np.sin(1.8 * theta_rad)**3)
        HL = HL_0 * psi
        HL = np.clip(HL, 0.01, 1.0)

        # Mixture density
        rho_m = rho_L * HL + rho_gas * (1 - HL)

        # Friction factor
        rho_ns = rho_L * lambda_L + rho_gas * (1 - lambda_L)
        mu_oil = 10**(0.43 + 8.33/API) * (T_F)**(-0.8) * 0.001  # Pa.s estimate
        mu_gas = 1e-5  # Pa.s
        mu_ns = mu_oil * lambda_L + mu_gas * (1 - lambda_L)

        Re = rho_ns * v_m * d_m / max(mu_ns, 1e-10)
        Re = max(Re, 100)
        fn = 1 / (2 * np.log10(Re / 4.5223 * np.log10(Re) - 3.8215))**2  # approx Moody
        fn = max(fn, 0.001)

        y = lambda_L / max(HL**2, 1e-10)
        if y > 1.0 and y < 1.2:
            S = np.log(2.2 * y - 1.2)
        elif 1e-4 < y <= 1.0 or y >= 1.2:
            ln_y = np.log(max(y, 1e-10))
            denom = -0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4
            if abs(denom) < 1e-10:
                S = 0
            else:
                S = ln_y / denom
        else:
            S = 0

        S = np.clip(S, -10, 10)
        fm = fn * np.exp(S)

        # Pressure gradient components
        dPdz_hydro = rho_m * g * np.sin(theta_rad)
        dPdz_fric = fm * rho_ns * v_m**2 / (2 * d_m)

        dPdz_total = dPdz_hydro + dPdz_fric  # Pa/m

        # Integrate over well depth
        delta_P_Pa = dPdz_total * H
        delta_P_bar = delta_P_Pa / 1e5

        return delta_P_bar

    except Exception:
        return np.nan


def _bb_C(lambda_L, NFr, params):
    """Beggs & Brill inclination correction coefficient."""
    d_coef, e_coef, f_coef, g_coef = params
    if d_coef == 0:
        return 0
    NLv = 1.0  # simplified - full NLv requires surface tension
    C = d_coef * lambda_L**e_coef * NFr**f_coef * NLv**g_coef
    return max(C, 0)


# Compute B&B predictions for all data points
print("\n  Computing Beggs & Brill predictions for all data points...")
bb_predictions = []
for _, row in d.iterrows():
    T_avg = (row.get('AVG_DOWNHOLE_TEMPERATURE', 100) + row.get('AVG_WHT_P', 60)) / 2
    dp_bb = beggs_brill_dP(row['q_oil'], row['q_gas'], row['q_wat'],
                           row['AVG_WHP_P'], T_avg)
    bb_predictions.append(dp_bb)

d_bb = d.copy()
d_bb['dP_BB'] = bb_predictions
d_bb = d_bb.dropna(subset=['dP_BB'])

# B&B metrics per well (using LOWO structure for fair comparison)
bb_results = []
for well in wells:
    well_data = d_bb[d_bb['Wellbore name'] == well]
    if len(well_data) == 0:
        continue
    yt = well_data[TARGET].values
    yp = well_data['dP_BB'].values
    valid = np.isfinite(yp) & np.isfinite(yt)
    if valid.sum() < 5:
        continue
    yt, yp = yt[valid], yp[valid]
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    r2 = r2_score(yt, yp)
    bb_results.append({
        'well': well, 'RMSE_BB': rmse, 'MAE_BB': mae, 'MAPE_BB': mape, 'R2_BB': r2
    })

bb_df = pd.DataFrame(bb_results)
print("\n  Beggs & Brill results (ASSUMED geometry):")
print(f"  {'Well':20s}  {'RMSE':>8s}  {'MAE':>8s}  {'MAPE%':>7s}  {'R2':>8s}")
for _, r in bb_df.iterrows():
    print(f"  {r['well']:20s}  {r['RMSE_BB']:8.2f}  {r['MAE_BB']:8.2f}  "
          f"{r['MAPE_BB']:7.2f}  {r['R2_BB']:8.3f}")

bb_df.to_csv(os.path.join(DATA_DIR, 'beggs_brill_results.csv'), index=False)

# Merge for comparison
comparison = lowo_df.merge(bb_df, left_on='test_well', right_on='well', how='left')
comparison.to_csv(os.path.join(DATA_DIR, 'rf_vs_bb_comparison.csv'), index=False)


# =============================================================================
# STAGE 8 - GENERATE ALL FIGURES
# =============================================================================
print("\n" + "=" * 72)
print("  STAGE 8/8: Generating publication-quality figures")
print("=" * 72)

# ---- FIGURE 1: LOWO Predicted vs Actual (scatter + bar chart) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes

for well, (yt, yp, wc) in lowo_preds.items():
    ax1.scatter(yt, yp, s=14, alpha=0.55, color=WELL_COLORS[well],
                label=short_well(well), edgecolors='none')

lims = [d[TARGET].min() - 10, d[TARGET].max() + 10]
ax1.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction', alpha=0.7)
ax1.set_xlabel('Actual dP (bar)')
ax1.set_ylabel('Predicted dP (bar)')
ax1.set_title('Leave-One-Well-Out: Predicted vs Actual')
ax1.legend(fontsize=8, loc='upper left')
ax1.set_xlim(lims); ax1.set_ylim(lims)

# Bar chart of per-fold R2 and RMSE
x = np.arange(len(lowo_df))
width = 0.35
bars1 = ax2.bar(x - width/2, lowo_df['R2'], width, color='#1565C0', label='R2', alpha=0.85)
ax2b = ax2.twinx()
bars2 = ax2b.bar(x + width/2, lowo_df['RMSE'], width, color='#B71C1C', label='RMSE (bar)', alpha=0.85)
ax2.axhline(0, color='gray', lw=0.8, linestyle='--')
ax2.set_xticks(x)
ax2.set_xticklabels([short_well(w) for w in lowo_df['test_well']], rotation=15)
ax2.set_ylabel('R2', color='#1565C0')
ax2b.set_ylabel('RMSE (bar)', color='#B71C1C')
ax2.set_title('Per-fold generalization (LOWO)')
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_lowo_predicted_vs_actual.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [1/8] fig1_lowo_predicted_vs_actual.png")


# ---- FIGURE 2: Chronological Split - Predicted vs Actual ----
n_wells_chrono = len(chrono_preds)
fig, axes = plt.subplots(1, min(n_wells_chrono, 5), figsize=(3.5 * min(n_wells_chrono, 5), 5),
                          squeeze=False)
axes = axes[0]

for i, (well, (yt, yp, dates)) in enumerate(chrono_preds.items()):
    if i >= 5:
        break
    ax = axes[i]
    dates_dt = pd.to_datetime(dates)
    ax.plot(dates_dt, yt, 'o-', ms=2, lw=0.8, color='#1565C0', label='Actual', alpha=0.7)
    ax.plot(dates_dt, yp, 's-', ms=2, lw=0.8, color='#E64A19', label='RF Predicted', alpha=0.7)
    ax.set_title(f'{short_well(well)}\nR2={chrono_df[chrono_df.test_well==well].R2.values[0]:.3f}',
                 fontsize=10)
    ax.set_xlabel('Date')
    if i == 0:
        ax.set_ylabel('dP (bar)')
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', rotation=45, labelsize=7)

plt.suptitle('Chronological Split: Last 20% of Each Well (Test Period)', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_chrono_split_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [2/8] fig2_chrono_split_timeseries.png")


# ---- FIGURE 3: Feature Importance (Gini) ----
fig, ax = plt.subplots(figsize=(8, 6))
gi = gini_imp.sort_values('gini_importance', ascending=True)
bars = ax.barh(gi['feature'], gi['gini_importance'], color='#2E7D32', alpha=0.85)
ax.set_xlabel('Feature Importance (Gini-based)')
ax.set_title('Which measured variables drive dP prediction')
for bar, val in zip(bars, gi['gini_importance']):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_feature_importance_gini.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [3/8] fig3_feature_importance_gini.png")


# ---- FIGURE 4: SHAP Beeswarm Plot ----
if HAS_SHAP and shap_values is not None:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES,
                      show=False, max_display=16)
    plt.title('SHAP Feature Importance (Beeswarm Plot)', fontsize=13)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_shap_beeswarm.png'),
                dpi=150, bbox_inches='tight')
    plt.close('all')
    print("  [4/8] fig4_shap_beeswarm.png")
else:
    print("  [4/8] SKIPPED (SHAP not available)")


# ---- FIGURE 5: RF vs Beggs & Brill Comparison ----
fig, ax = plt.subplots(figsize=(10, 6))
merged = comparison.dropna(subset=['RMSE_BB'])
x = np.arange(len(merged))
width = 0.35
ax.bar(x - width/2, merged['RMSE'], width, color='#1565C0', label='Random Forest (LOWO)', alpha=0.85)
ax.bar(x + width/2, merged['RMSE_BB'], width, color='#B71C1C', label='Beggs & Brill (assumed geom.)', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([short_well(w) for w in merged['test_well']], rotation=15)
ax.set_ylabel('RMSE (bar)')
ax.set_title('RF vs Beggs & Brill -- RMSE by held-out well\n*B&B uses assumed tubing ID/depth/inclination -- not measured')
ax.legend()

# Add value labels
for i, (rf_v, bb_v) in enumerate(zip(merged['RMSE'], merged['RMSE_BB'])):
    ax.text(i - width/2, rf_v + 1, f'{rf_v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + width/2, bb_v + 1, f'{bb_v:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig5_rf_vs_bb.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [5/8] fig5_rf_vs_bb.png")


# ---- FIGURE 6: VLP Curve (dP vs q_liq, colored by WC) ----
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(d['q_liq'], d['delta_P'], c=d['WC'], cmap='RdYlBu_r',
                s=12, alpha=0.6, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Water Cut (fraction)')
ax.set_xlabel('Liquid Flow Rate (Sm3/d)')
ax.set_ylabel('Wellbore Pressure Drop, dP (bar)')
ax.set_title('VLP Relationship: Pressure Drop vs Liquid Rate\nColored by Water Cut')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig6_vlp_curve_by_wc.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [6/8] fig6_vlp_curve_by_wc.png")


# ---- FIGURE 7: Residual Analysis ----
# Use pooled split for residual analysis
residuals = yt_pool - yp_pool

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 7a: Residuals vs Predicted
ax = axes[0, 0]
ax.scatter(yp_pool, residuals, s=10, alpha=0.4, color='#1565C0', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Predicted dP (bar)')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_title('Residuals vs Predicted')

# 7b: Residual histogram
ax = axes[0, 1]
ax.hist(residuals, bins=40, color='#388E3C', alpha=0.75, edgecolor='white')
ax.axvline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Residual (bar)')
ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')

# 7c: Residuals vs Water Cut
wc_test = X_test['WC'].values
ax = axes[1, 0]
ax.scatter(wc_test, residuals, s=10, alpha=0.4, color='#E64A19', edgecolors='none')
ax.axhline(0, color='red', linestyle='--', lw=1.5)
ax.set_xlabel('Water Cut')
ax.set_ylabel('Residual (bar)')
ax.set_title('Residuals vs Water Cut')

# 7d: Absolute error vs GOR
gor_test = X_test['GOR'].values
ax = axes[1, 1]
ax.scatter(gor_test, np.abs(residuals), s=10, alpha=0.4, color='#7B1FA2', edgecolors='none')
ax.set_xlabel('GOR (Sm3/Sm3)')
ax.set_ylabel('Absolute Error (bar)')
ax.set_title('Absolute Error vs GOR')

plt.suptitle('Residual Analysis (Pooled 80/20 Split)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig7_residual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [7/8] fig7_residual_analysis.png")


# ---- FIGURE 8: Validation Strategy Comparison ----
fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['LOWO\n(cross-well)', 'Chronological\n(within-well)', 'Pooled\n(random 80/20)']
r2_values = [lowo_df['R2'].mean(), chrono_df['R2'].mean(), pool_r2]
rmse_values = [lowo_df['RMSE'].mean(), chrono_df['RMSE'].mean(), pool_rmse]
r2_stds = [lowo_df['R2'].std(), chrono_df['R2'].std(), 0]

x = np.arange(len(strategies))
width = 0.35

bars1 = ax.bar(x - width/2, r2_values, width, color='#1565C0', label='Mean R2', alpha=0.85,
               yerr=r2_stds, capsize=5)
ax_r = ax.twinx()
bars2 = ax_r.bar(x + width/2, rmse_values, width, color='#B71C1C', label='Mean RMSE (bar)', alpha=0.85)

ax.axhline(0, color='gray', lw=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylabel('R2', color='#1565C0')
ax_r.set_ylabel('RMSE (bar)', color='#B71C1C')
ax.set_title('Comparison of Three Validation Strategies')

# Value labels
for bar, val in zip(bars1, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, max(val, 0) + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, rmse_values):
    ax_r.text(bar.get_x() + bar.get_width()/2, val + 0.3,
              f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_r.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='upper left')

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig8_validation_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("  [8/8] fig8_validation_comparison.png")


# =============================================================================
# SAVE MODEL-READY DATA, TRAINED MODEL, AND SUMMARY TABLE
# =============================================================================
d.to_csv(os.path.join(DATA_DIR, 'modelready_features.csv'), index=False)

# Save the trained model for Streamlit app
import pickle

model_package = {
    'model': rf_full,
    'features': FEATURES,
    'target': TARGET,
    'best_params': best_params,
    'training_stats': {
        'n_samples': len(d),
        'n_features': len(FEATURES),
        'target_mean': d[TARGET].mean(),
        'target_std': d[TARGET].std(),
    },
}

model_path = os.path.join(DATA_DIR, 'rf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_package, f)
print(f"\n  Saved trained model to {model_path}")

# Create a comprehensive summary table
print("\n" + "=" * 72)
print("  COMPREHENSIVE RESULTS SUMMARY")
print("=" * 72)

summary_table = pd.DataFrame([
    {'Strategy': 'LOWO (cross-well)', 'Mean_RMSE': lowo_df.RMSE.mean(),
     'Std_RMSE': lowo_df.RMSE.std(), 'Mean_MAE': lowo_df.MAE.mean(),
     'Mean_MAPE': lowo_df.MAPE.mean(), 'Mean_R2': lowo_df.R2.mean(),
     'Std_R2': lowo_df.R2.std(), 'Best_R2': lowo_df.R2.max(),
     'Worst_R2': lowo_df.R2.min()},
    {'Strategy': 'Chrono (within-well)', 'Mean_RMSE': chrono_df.RMSE.mean(),
     'Std_RMSE': chrono_df.RMSE.std(), 'Mean_MAE': chrono_df.MAE.mean(),
     'Mean_MAPE': chrono_df.MAPE.mean(), 'Mean_R2': chrono_df.R2.mean(),
     'Std_R2': chrono_df.R2.std(), 'Best_R2': chrono_df.R2.max(),
     'Worst_R2': chrono_df.R2.min()},
    {'Strategy': 'Pooled (random)', 'Mean_RMSE': pool_rmse,
     'Std_RMSE': 0, 'Mean_MAE': pool_mae,
     'Mean_MAPE': pool_mape, 'Mean_R2': pool_r2,
     'Std_R2': 0, 'Best_R2': pool_r2,
     'Worst_R2': pool_r2},
])
summary_table.to_csv(os.path.join(DATA_DIR, 'validation_summary.csv'), index=False)

print(summary_table.to_string(index=False))
print(f"\n  Best RF hyperparameters: {best_params}")
print(f"  Total features used: {len(FEATURES)}")
print(f"  Total steady-state data points: {len(d)}")
print(f"  Wells: {len(wells)} ({', '.join([short_well(w) for w in wells])})")

print("\n" + "=" * 72)
print("  OUTPUT FILES")
print("=" * 72)
outputs = {
    'data/lowo_results.csv': 'Leave-One-Well-Out cross-validation results',
    'data/chrono_split_results.csv': 'Chronological within-well split results',
    'data/pooled_split_results.csv': 'Pooled random split results',
    'data/validation_summary.csv': 'All 3 strategies compared',
    'data/classification_metrics.csv': 'Accuracy, Precision, Recall, F1 per regime class',
    'data/feature_importance_gini.csv': 'Gini-based feature importance',
    'data/beggs_brill_results.csv': 'Beggs & Brill benchmark (ASSUMED geometry)',
    'data/rf_vs_bb_comparison.csv': 'RF vs B&B side-by-side comparison',
    'data/modelready_features.csv': 'Cleaned, feature-engineered dataset',
    'data/rf_model.pkl': 'Trained RF model (for Streamlit app)',
    'figures/fig1_lowo_predicted_vs_actual.png': 'LOWO scatter + per-fold bars',
    'figures/fig2_chrono_split_timeseries.png': 'Chronological split time series',
    'figures/fig3_feature_importance_gini.png': 'Gini feature importance',
    'figures/fig4_shap_beeswarm.png': 'SHAP beeswarm plot',
    'figures/fig5_rf_vs_bb.png': 'RF vs Beggs & Brill comparison',
    'figures/fig6_vlp_curve_by_wc.png': 'VLP curve colored by water cut',
    'figures/fig7_residual_analysis.png': '4-panel residual analysis',
    'figures/fig8_validation_comparison.png': '3-strategy validation comparison',
}
for f, desc in outputs.items():
    path = os.path.join(BASE_DIR, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = f'{size:>10,} bytes' if exists else 'NOT FOUND'
    print(f"  {'OK' if exists else 'XX'}  {f:50s}  {status}  {desc}")

if HAS_SHAP:
    print("\n  data/feature_importance_shap.csv also generated.")

print("\n" + "=" * 72)
print("  PIPELINE COMPLETE")
print("  Every number above came from this single run.")
print("  Cite these numbers in your report, not earlier drafts.")
print("=" * 72)
