"""
VLP RANDOM FOREST MODEL — REPRODUCIBLE PIPELINE
Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow Prediction

Run this ONE script top to bottom. Every number in your report should trace back
to this file so you can defend it if questioned. Do not hand-copy numbers between
runs — re-run this and use whatever it prints.

INPUT REQUIRED: volve_welldata.csv in the same folder (or edit INPUT_PATH below)

────────────────────────────────────────────────────────────────────────────
HONEST ASSUMPTIONS — read before you present this. Your dataset does NOT
contain tubing diameter, well depth, deviation survey, or fluid PVT reports.
Anything requiring those (the Beggs & Brill benchmark) uses placeholder,
representative values marked ASSUMED below. Get the real values from Volve's
public completion reports before final submission, or explicitly disclose
the assumption in your methodology chapter. The core Random Forest model
does NOT depend on these assumptions — it uses only directly measured columns.
────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_PATH = 'volve_welldata.csv'
OUTPUT_DIR = '.'

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD AND FILTER TO USABLE PRODUCER WELLS
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(INPUT_PATH)
op = df[df['WELL_TYPE'] == 'OP'].copy()

# Only wells with any downhole pressure gauge data at all.
# F-4 AH is 100% a water injector (WELL_TYPE=WI everywhere) — cannot be used.
# F-5 AH has 144 producer rows but ZERO valid downhole pressure — cannot be used.
# Verify this yourself: df[df['Wellbore name']=='15/9-F-4 AH']['WELL_TYPE'].value_counts()
USABLE_WELLS = ['15/9-F-1 C', '15/9-F-11 H', '15/9-F-12 H', '15/9-F-14 H', '15/9-F-15 D']
op = op[op['Wellbore name'].isin(USABLE_WELLS)].copy()

NUM_COLS = ['ON_STREAM_HRS','AVG_DOWNHOLE_PRESSURE','AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING','AVG_CHOKE_SIZE_P','AVG_WHP_P','AVG_WHT_P',
            'BORE_OIL_VOL','BORE_GAS_VOL','BORE_WAT_VOL']
for c in NUM_COLS:
    op[c] = pd.to_numeric(op[c], errors='coerce')

mask = ((op['AVG_DOWNHOLE_PRESSURE']>0) & (op['AVG_WHP_P']>0) &
        (op['ON_STREAM_HRS']>0) & (op['BORE_OIL_VOL']>0) &
        (op['AVG_DP_TUBING'].notna()) & (op['AVG_DP_TUBING']>0))
clean = op[mask].copy()
print(f"[1/5] Valid producer rows: {len(clean)} across {clean['Wellbore name'].nunique()} wells")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — FEATURE ENGINEERING (real, measured/derived quantities only)
# ═══════════════════════════════════════════════════════════════════════════
clean['Date of Production'] = pd.to_datetime(clean['Date of Production'], format='mixed')
clean = clean.sort_values(['Wellbore name','Date of Production']).reset_index(drop=True)

clean['q_oil'] = clean['BORE_OIL_VOL']*24.0/clean['ON_STREAM_HRS']
clean['q_gas'] = clean['BORE_GAS_VOL']*24.0/clean['ON_STREAM_HRS']
clean['q_wat'] = clean['BORE_WAT_VOL']*24.0/clean['ON_STREAM_HRS']
clean['q_liq'] = clean['q_oil'] + clean['q_wat']
clean['WC']    = clean['q_wat']/clean['q_liq'].replace(0,np.nan)
clean['GOR']   = clean['q_gas']/clean['q_oil'].replace(0,np.nan)
clean['delta_P'] = clean['AVG_DOWNHOLE_PRESSURE'] - clean['AVG_WHP_P']  # bar — the real target

clean = clean[(clean['WC']>=0)&(clean['WC']<=1)&(clean['GOR']>0)&(clean['GOR']<5000)&(clean['q_liq']>0)]

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — STEADY-STATE FILTER (reproducible; threshold justified, not arbitrary)
# ═══════════════════════════════════════════════════════════════════════════
# 7-day rolling coefficient of variation on liquid rate, and rolling std of dP.
# NOTE: a strict 10-bar / low-CV threshold (as sometimes quoted in early drafts
# of this project) keeps <1% of the data — it is NOT achievable on this dataset
# and should not be claimed. The thresholds below were chosen by checking the
# actual distribution (median CV ~0.28, median rolling std(dP) ~23 bar) and
# picking a cut that keeps a representative sample from every well.
g = clean.groupby('Wellbore name', group_keys=False)
clean['roll_std_q']  = g['q_liq'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['roll_mean_q'] = g['q_liq'].rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
clean['roll_std_dP'] = g['delta_P'].rolling(7, min_periods=3).std().reset_index(level=0, drop=True)
clean['cv_q'] = clean['roll_std_q']/clean['roll_mean_q'].replace(0,np.nan)

CV_THRESHOLD, DP_STD_THRESHOLD = 0.35, 30.0
steady_mask = (clean['cv_q'] < CV_THRESHOLD) & (clean['roll_std_dP'] < DP_STD_THRESHOLD) & clean['cv_q'].notna()
steady = clean[steady_mask].copy()
print(f"[2/5] Steady-state rows: {len(steady)} of {len(clean)} ({len(steady)/len(clean)*100:.1f}%)")
print(steady['Wellbore name'].value_counts().to_string())

FEATURES = ['q_oil','q_gas','q_wat','q_liq','WC','GOR',
            'AVG_WHP_P','AVG_WHT_P','AVG_DOWNHOLE_TEMPERATURE',
            'AVG_CHOKE_SIZE_P','ON_STREAM_HRS']
TARGET = 'delta_P'
d = steady[['Wellbore name']+FEATURES+[TARGET]].dropna()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — LEAVE-ONE-WELL-OUT CROSS VALIDATION (the real generalization test)
# ═══════════════════════════════════════════════════════════════════════════
# A single arbitrary train/test split (e.g. "always test on F-1C") is fragile
# and cherry-pickable with only 5 wells. Report ALL 5 folds, not one.
wells = d['Wellbore name'].unique()
results, preds_all = [], {}

for test_well in wells:
    train = d[d['Wellbore name'] != test_well]
    test  = d[d['Wellbore name'] == test_well]
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(train[FEATURES], train[TARGET])
    yp = rf.predict(test[FEATURES]); yt = test[TARGET].values
    rmse = np.sqrt(mean_squared_error(yt,yp)); mae = mean_absolute_error(yt,yp)
    mape = np.mean(np.abs((yt-yp)/yt))*100
    ss_res=np.sum((yt-yp)**2); ss_tot=np.sum((yt-yt.mean())**2); r2=1-ss_res/ss_tot
    results.append(dict(test_well=test_well, n_test=len(test), n_train=len(train),
                         RMSE=rmse, MAE=mae, MAPE=mape, R2=r2))
    preds_all[test_well] = (yt, yp)

res_df = pd.DataFrame(results)
print("\n[3/5] Leave-one-well-out results (this is your primary validation table):")
print(res_df.to_string(index=False))
print(f"\nMean across folds:  RMSE={res_df.RMSE.mean():.2f}  MAPE={res_df.MAPE.mean():.2f}%  R2={res_df.R2.mean():.3f}")
print(f"Std across folds:   RMSE={res_df.RMSE.std():.2f}  MAPE={res_df.MAPE.std():.2f}%  R2={res_df.R2.std():.3f}")
print("^ report BOTH mean and std — the std is the honest measure of how much this varies by well.\n")

res_df.to_csv(f'{OUTPUT_DIR}/lowo_results.csv', index=False)

# Feature importance from a model trained on everything (for interpretation only,
# never for the reported accuracy numbers — those come from the LOWO folds above)
rf_full = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features='sqrt',
                                 random_state=42, n_jobs=-1)
rf_full.fit(d[FEATURES], d[TARGET])
imp = pd.Series(rf_full.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("[4/5] Feature importance (trained on all wells, for interpretation):")
print(imp.to_string())
imp.to_csv(f'{OUTPUT_DIR}/feature_importance.csv')

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13,5.5))
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0,1,len(wells)))
for (well,(yt,yp)), c in zip(preds_all.items(), colors):
    ax.scatter(yt, yp, s=14, alpha=0.6, color=c, label=well)
lims=[d[TARGET].min()-10, d[TARGET].max()+10]
ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual ΔP (bar)'); ax.set_ylabel('Predicted ΔP (bar)')
ax.set_title('Leave-One-Well-Out: Predicted vs Actual')
ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)

ax2 = axes[1]; x=np.arange(len(res_df)); ax2b = ax2.twinx()
ax2.bar(x-0.18, res_df['R2'], width=0.36, color='#1565C0', label='R²')
ax2b.bar(x+0.18, res_df['RMSE'], width=0.36, color='#B71C1C', label='RMSE (bar)')
ax2.axhline(0, color='gray', lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels([w.replace('15/9-','') for w in res_df['test_well']], rotation=15)
ax2.set_ylabel('R²', color='#1565C0'); ax2b.set_ylabel('RMSE (bar)', color='#B71C1C')
ax2.set_title('Per-fold generalization')
h1,l1=ax2.get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels()
ax2.legend(h1+h2,l1+l2,fontsize=8,loc='lower right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_lowo_predicted_vs_actual.png', dpi=150, bbox_inches='tight')

fig2, ax = plt.subplots(figsize=(8,5.5))
ax.barh(imp.index[::-1], imp.values[::-1], color='#2E7D32')
ax.set_xlabel('Random Forest feature importance (Gini-based)')
ax.set_title('Which measured variables drive ΔP prediction')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_feature_importance.png', dpi=150, bbox_inches='tight')

print("\n[5/5] Saved: lowo_results.csv, feature_importance.csv, fig1_*.png, fig2_*.png")
print("Pipeline complete. Every number above came from this single run — cite these, not earlier drafts.")
