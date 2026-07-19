"""
DIAGNOSTIC SCRIPT - Understand the data before improving the model.
Prints per-well statistics to identify why F-15D and F-12H fail.
"""
import pandas as pd
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

INPUT_PATH = 'data/volve_welldata_raw.csv'
df = pd.read_csv(INPUT_PATH)

# --- Basic dataset shape ---
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Total rows: {len(df)}")
print(f"\nWELL_TYPE distribution:")
print(df['WELL_TYPE'].value_counts().to_string())

# --- Filter to usable producer rows ---
op = df[df['WELL_TYPE'] == 'OP'].copy()
USABLE_WELLS = ['15/9-F-1 C', '15/9-F-11 H', '15/9-F-12 H', '15/9-F-14 H', '15/9-F-15 D']
op = op[op['Wellbore name'].isin(USABLE_WELLS)].copy()

NUM_COLS = ['ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_DP_TUBING', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'AVG_WHT_P',
            'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
for c in NUM_COLS:
    op[c] = pd.to_numeric(op[c], errors='coerce')

mask = ((op['AVG_DOWNHOLE_PRESSURE'] > 0) & (op['AVG_WHP_P'] > 0) &
        (op['ON_STREAM_HRS'] > 0) & (op['BORE_OIL_VOL'] > 0) &
        (op['AVG_DP_TUBING'].notna()) & (op['AVG_DP_TUBING'] > 0))
clean = op[mask].copy()

clean['q_oil'] = clean['BORE_OIL_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_gas'] = clean['BORE_GAS_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_wat'] = clean['BORE_WAT_VOL'] * 24.0 / clean['ON_STREAM_HRS']
clean['q_liq'] = clean['q_oil'] + clean['q_wat']
clean['WC'] = clean['q_wat'] / clean['q_liq'].replace(0, np.nan)
clean['GOR'] = clean['q_gas'] / clean['q_oil'].replace(0, np.nan)
clean['delta_P'] = clean['AVG_DOWNHOLE_PRESSURE'] - clean['AVG_WHP_P']

clean = clean[(clean['WC'] >= 0) & (clean['WC'] <= 1) & (clean['GOR'] > 0) &
              (clean['GOR'] < 5000) & (clean['q_liq'] > 0)]

print(f"\nFiltered to {len(clean)} clean producer rows across {clean['Wellbore name'].nunique()} wells")

# --- Per-well detailed statistics ---
print("\n" + "=" * 70)
print("PER-WELL STATISTICS")
print("=" * 70)

key_vars = ['delta_P', 'q_oil', 'q_wat', 'q_liq', 'WC', 'GOR',
            'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_PRESSURE',
            'AVG_DOWNHOLE_TEMPERATURE', 'AVG_CHOKE_SIZE_P']

for well in USABLE_WELLS:
    w = clean[clean['Wellbore name'] == well]
    print(f"\n--- WELL: {well}  ({len(w)} rows) ---")
    for v in key_vars:
        vals = w[v].dropna()
        if len(vals) > 0:
            print(f"  {v:35s}  min={vals.min():10.2f}  mean={vals.mean():10.2f}  "
                  f"max={vals.max():10.2f}  std={vals.std():10.2f}")

# --- delta_P range overlap analysis ---
print("\n" + "=" * 70)
print("DELTA_P RANGE OVERLAP - Why cross-well prediction fails")
print("=" * 70)
for well in USABLE_WELLS:
    w = clean[clean['Wellbore name'] == well]
    dp = w['delta_P']
    print(f"  {well:20s}  dP range: [{dp.min():.1f}, {dp.max():.1f}] bar  "
          f"mean={dp.mean():.1f}  std={dp.std():.1f}  n={len(w)}")

# --- WC range per well ---
print("\n" + "=" * 70)
print("WATER CUT RANGE PER WELL")
print("=" * 70)
for well in USABLE_WELLS:
    w = clean[clean['Wellbore name'] == well]
    wc = w['WC']
    print(f"  {well:20s}  WC range: [{wc.min():.3f}, {wc.max():.3f}]  mean={wc.mean():.3f}")

# --- Cross-correlation check ---
print("\n" + "=" * 70)
print("FEATURE CORRELATIONS WITH delta_P (whole dataset)")
print("=" * 70)
for v in key_vars:
    if v != 'delta_P':
        corr = clean[['delta_P', v]].dropna().corr().iloc[0, 1]
        print(f"  {v:35s}  r = {corr:+.4f}")

# --- Per-well correlations with delta_P ---
print("\n" + "=" * 70)
print("PER-WELL CORRELATIONS WITH delta_P (key features)")
print("=" * 70)
focus_feats = ['WC', 'q_liq', 'GOR', 'AVG_WHP_P', 'AVG_DOWNHOLE_TEMPERATURE']
header = f"  {'Well':20s}" + "".join([f"  {f:>12s}" for f in focus_feats])
print(header)
for well in USABLE_WELLS:
    w = clean[clean['Wellbore name'] == well]
    row = f"  {well:20s}"
    for f in focus_feats:
        corr = w[['delta_P', f]].dropna().corr().iloc[0, 1]
        row += f"  {corr:>+12.3f}"
    print(row)

# --- Operating regime summary ---
print("\n" + "=" * 70)
print("OPERATING REGIME SUMMARY")
print("=" * 70)
summary_rows = []
for well in USABLE_WELLS:
    w = clean[clean['Wellbore name'] == well]
    summary_rows.append({
        'well': well.replace('15/9-', ''),
        'n': len(w),
        'dP_mean': round(w['delta_P'].mean(), 1),
        'dP_std': round(w['delta_P'].std(), 1),
        'WC_mean': round(w['WC'].mean(), 3),
        'GOR_mean': round(w['GOR'].mean(), 1),
        'q_liq_mean': round(w['q_liq'].mean(), 1),
        'WHP_mean': round(w['AVG_WHP_P'].mean(), 1),
    })
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
