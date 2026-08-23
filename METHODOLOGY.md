# 📐 Methodology

Technical documentation of the data processing pipeline, feature engineering,
model selection, validation strategies, and benchmarking approach.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Data Source & Preprocessing](#data-source--preprocessing)
- [Feature Engineering](#feature-engineering)
- [Steady-State Filter](#steady-state-filter)
- [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Validation Strategies](#validation-strategies)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Beggs & Brill Benchmark](#beggs--brill-benchmark)
- [Assumptions & Simplifications](#assumptions--simplifications)

---

## Problem Statement

In petroleum engineering, **Vertical Lift Performance (VLP)** models predict the pressure
drop that occurs as fluids travel from the reservoir up through the wellbore to the surface.
This pressure drop, ΔP = P_wf − P_wh (bottomhole flowing pressure minus wellhead pressure),
is a function of:

- Multiphase flow regime (oil, gas, water flowing simultaneously)
- Flow rates and fluid composition
- Well geometry (tubing diameter, depth, inclination)
- Temperature and pressure profiles

Classical empirical correlations (Beggs & Brill 1973, Hagedorn & Brown 1965, etc.) require
detailed well geometry and fluid PVT data. When these are unavailable or inaccurate, a
data-driven approach trained on actual production measurements can provide a superior alternative.

**Objective:** Develop a Random Forest regression model that predicts ΔP from directly
measured production data, validate it rigorously, and compare it against the Beggs & Brill
(1973) correlation.

---

## Data Source & Preprocessing

### Source

The Volve field dataset (Equinor, 2018) is a publicly released dataset from a decommissioned
North Sea oil field. It contains daily production data for multiple wells from 2008–2016.

### Filtering Steps

The raw data undergoes three filtering stages:

#### Filter 1: Well Type & Well Selection

```
15,634 raw rows → 7,029 producer rows for usable wells
```

- Keep only rows with `WELL_TYPE = 'OP'` (oil producer)
- Keep only the 5 wells with valid downhole pressure gauge (PDG) data:
  - **F-1C**, **F-11H**, **F-12H**, **F-14H**, **F-15D**
- Wells explicitly excluded and why:
  - F-4 AH: 100% water injector (WI) — no producer data
  - F-5 AH: 144 producer rows but zero valid downhole pressure readings

#### Filter 2: Data Validity

```
7,029 → ~4,176 valid rows
```

All of the following must be true:
- `AVG_DOWNHOLE_PRESSURE > 0`
- `AVG_WHP_P > 0`
- `ON_STREAM_HRS > 0`
- `BORE_OIL_VOL > 0`
- `AVG_DP_TUBING` is not null and > 0

#### Filter 3: Physical Consistency

```
~4,176 → ~5,630 model-ready rows (after feature engineering adds valid rows)
```

- Water cut: 0 ≤ WC ≤ 1
- GOR: 0 < GOR < 5,000 Sm³/Sm³
- Liquid rate: q_liq > 0
- Pressure drop: ΔP > 0

---

## Feature Engineering

Sixteen features are engineered from the raw measured columns. Each feature is chosen
for its physical relevance to multiphase wellbore flow, not as an arbitrary statistical input.

### Flow Rates (Normalized by On-Stream Hours)

Raw daily volumes are converted to instantaneous rates. This correction is essential because
a well that produced for only 12 hours should not appear to have half the rate of a full-day
producer — the instantaneous flow rate determines the actual hydraulic conditions in the tubing.

| Feature | Formula | Unit | Physical Justification |
|---------|---------|------|----------------------|
| `q_oil` | `BORE_OIL_VOL × 24 / ON_STREAM_HRS` | Sm³/d | Instantaneous oil rate — determines oil velocity in tubing |
| `q_gas` | `BORE_GAS_VOL × 24 / ON_STREAM_HRS` | Sm³/d | Instantaneous gas rate — controls gas void fraction |
| `q_wat` | `BORE_WAT_VOL × 24 / ON_STREAM_HRS` | Sm³/d | Instantaneous water rate — increases mixture density |
| `q_liq` | `q_oil + q_wat` | Sm³/d | Total liquid throughput — determines mixture velocity, which directly affects friction gradient (dP/dz_friction ∝ v²) |

### Compositional Features (Account for ~78% of Model Importance)

These are the most physically meaningful features. In multiphase flow, the **ratio** of gas
to liquid is more important than absolute rates, because it determines the flow regime
(bubble, slug, annular, mist) and the gas void fraction, which in turn controls the
hydrostatic pressure gradient.

| Feature | Formula | Physical Justification |
|---------|---------|----------------------|
| `WC` | `q_wat / q_liq` | **Water cut** — drives mixture density. Water (ρ≈1020 kg/m³) is much denser than oil (ρ≈850 kg/m³). Higher WC = heavier fluid column = higher hydrostatic ΔP. Also triggers transitions between oil-continuous and water-continuous flow regimes. |
| `GOR` | `q_gas / q_oil` | **Gas-oil ratio** — controls solution gas behavior. At pressures above the bubble point, gas dissolves in oil; below, free gas appears. GOR determines how much gas is available to lighten the fluid column (gas-lift effect). |
| `GLR` | `q_gas / q_liq` | **Gas-liquid ratio** — a more general indicator than GOR for wells with significant water production. Directly governs gas void fraction, which is the key parameter in all multiphase flow correlations (including Beggs & Brill). |

### Log-Transformed Rates

| Feature | Formula | Physical Justification |
|---------|---------|----------------------|
| `log_q_liq` | `ln(1 + q_liq)` | Gas rates can vary by 100x across wells while oil rates vary by 5-10x. Log transformation compresses this dynamic range so that Random Forest tree splits are effective across the entire range, rather than being dominated by a few high-rate points. |
| `log_q_oil` | `ln(1 + q_oil)` | Same rationale — improves split efficiency for tree-based models. |
| `log_q_gas` | `ln(1 + q_gas)` | Same rationale — particularly important for gas, which has the widest range. |

### Temperature Features

| Feature | Formula | Physical Justification |
|---------|---------|----------------------|
| `dT` | `T_downhole − T_wellhead` | **Temperature gradient** — a proxy for the geothermal gradient and heat transfer along the wellbore. Temperature affects oil viscosity (exponentially), gas solubility, wax/asphaltene deposition risk, and fluid density. A large dT indicates significant property changes between bottom and top of well. |

### Directly Measured Features (Passed Through)

| Feature | Source Column | Physical Justification |
|---------|--------------|----------------------|
| `AVG_WHP_P` | Wellhead pressure (bar) | Surface backpressure — compresses gas at wellhead conditions, affecting void fraction and density. This is a standard VLP input. |
| `AVG_WHT_P` | Wellhead temperature (°C) | Surface temperature — affects fluid PVT properties at wellhead. |
| `AVG_DOWNHOLE_TEMPERATURE` | Downhole temperature (°C) | Reservoir/gauge temperature — controls viscosity, gas solubility (Rs), oil formation volume factor (Bo). |
| `AVG_CHOKE_SIZE_P` | Choke opening (%) | Choke opening controls the flow rate and creates a surface pressure drop. |
| `ON_STREAM_HRS` | Hours on production | After rate normalization, captures whether the well operated in transient start-up mode or steady state. |

### Note: P_ratio is Computed but NOT Used

`P_ratio = P_wf / P_wh` is computed during feature engineering but is **excluded from the
final feature set** because it directly contains the target variable (P_wf is used in computing
ΔP = P_wf − P_wh). Including P_ratio would constitute **data leakage**.

### Target Variable

```
delta_P = AVG_DOWNHOLE_PRESSURE − AVG_WHP_P    (bar)
```

This is the wellbore pressure drop — the quantity the model predicts. It is verified against
the independently recorded `AVG_DP_TUBING` column (correlation = 0.9997).

---

## Data Leakage Audit

> ⚠️ **This is the most important quality assurance check in the project.**

Because the target variable ΔP = P_wf − P_wh is computed from two measured pressures, we
must rigorously verify that no feature directly or indirectly contains the answer.

### Automated Verification

The pipeline performs an explicit check at runtime. If any of the following columns appear
in the feature list, the pipeline halts with an error:

| Column | Description | In Features? | Why It Would Be Leakage |
|--------|-------------|:------------:|------------------------|
| `AVG_DP_TUBING` | Measured tubing dP | ❌ NO | This IS the target (correlation 0.9997 with delta_P) |
| `AVG_DOWNHOLE_PRESSURE` | Bottomhole pressure (P_wf) | ❌ NO | Directly used to compute delta_P |
| `P_ratio` | P_wf / P_wh | ❌ NO | Contains P_wf, which is part of delta_P |
| `delta_P` | The target variable itself | ❌ NO | This would make R²=1.0 trivially |

### AVG_WHP_P in Both Features and Target Formula

**`AVG_WHP_P`** is NOT leakage because:
1. It is measured by an **independent** surface pressure gauge
2. In deployment, WHP would always be **available** as an input
3. It is a **standard input** to all VLP correlations (including Beggs & Brill)
4. Removing it would cripple the model by removing physically necessary information

### Verdict: NO DATA LEAKAGE DETECTED

---

## Steady-State Filter

Transient data (well start-ups, shut-ins, choke changes) is filtered using a 7-day rolling
window approach:

| Parameter | Threshold | Purpose |
|-----------|:---------:|---------|
| CV of q_liq (coefficient of variation) | < 0.35 | Removes periods of rapidly changing flow rate |
| σ(ΔP) (standard deviation of pressure drop) | < 30 bar | Removes pressure transient periods |
| Minimum window | 3 days | Ensures enough data points for statistics |

**Typical retention rate:** ~95% of data survives (most data is steady-state).

The thresholds were determined empirically by examining the distribution of CV and σ(ΔP)
across all wells using `code/diagnose_data.py`.

---

## Model Selection

### Why Random Forest?

| Property | Advantage for VLP Prediction |
|----------|------------------------------|
| Non-parametric | No assumed functional form for ΔP vs flow rates |
| Handles nonlinearity | Multiphase flow is inherently nonlinear |
| Feature importance built-in | Gini importance is free with training |
| Robust to outliers | Individual trees vote, reducing outlier impact |
| No feature scaling needed | Tree splits are scale-invariant |
| Ensemble averaging | Reduces variance compared to single decision tree |

### Model Specification

```python
RandomForestRegressor(
    n_estimators=...,       # Tuned via GridSearchCV
    min_samples_leaf=...,   # Tuned via GridSearchCV
    max_features=...,       # Tuned via GridSearchCV
    max_depth=None,         # Unbounded (let trees grow fully)
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores
)
```

---

## Hyperparameter Tuning

### Grid Search Configuration

| Hyperparameter | Values Tested | Purpose |
|---------------|---------------|---------|
| `n_estimators` | [300, 500] | Number of trees in the forest |
| `min_samples_leaf` | [1, 3] | Minimum samples at a leaf node (controls overfitting) |
| `max_features` | ['sqrt', 0.5] | Features considered per split |

**Total combinations:** 2 × 2 × 2 = 8 configurations

**Cross-validation:** 3-fold CV on 80% of the data, scored by R².

**Tuning data:** 80% random sample (stratified), keeping 20% completely unseen.

The best parameters from GridSearchCV are used for all subsequent validation.

---

## Validation Strategies

Three distinct validation strategies provide different perspectives on model performance:

### Strategy A: Leave-One-Well-Out (LOWO) Cross-Validation

```
For each of the 5 wells:
    Train on the other 4 wells
    Test on the held-out well
    Record RMSE, MAE, MAPE, R²
```

**What it measures:** Can the model predict a **completely new well** it has never seen?

**Why it matters:** This is the strictest test. In practice, you might want to use the model
on a new well where no downhole data exists yet.

**Expected behavior:** High variance across folds. Wells with unique operating conditions
(e.g., F-15D with very narrow ΔP range) will score poorly because the training set doesn't
contain similar data.

### Strategy B: Chronological Within-Well Split

```
For each well:
    Sort data by date
    Train on: ALL other wells + first 80% of this well's timeline
    Test on: last 20% of this well's timeline
    Record metrics
```

**What it measures:** Can the model predict the **future** of a well it has partial history for?

**Why it matters:** This is the most realistic deployment scenario — you have some historical
data from a well and want to predict its future behavior.

### Strategy C: Pooled Random Split

```
Combine all wells
Randomly split 80% train / 20% test
Train and evaluate
```

**What it measures:** Overall **interpolation ability** across the dataset.

**Why it matters:** This is the least strict test but establishes a performance ceiling.
High performance here (R²=0.976) confirms the model has learned meaningful patterns,
not just noise.

### Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| RMSE | √(mean(yᵢ − ŷᵢ)²) | Average prediction error in bar (same units as ΔP) |
| MAE | mean(\|yᵢ − ŷᵢ\|) | Average absolute error in bar |
| MAPE | mean(\|yᵢ − ŷᵢ\| / yᵢ) × 100 | Percentage error relative to actual value |
| R² | 1 − SS_res / SS_tot | Proportion of variance explained (1.0 = perfect) |

---

## Feature Importance Analysis

Two complementary methods are used:

### Gini Importance (Mean Decrease in Impurity)

Built into scikit-learn's `RandomForestRegressor`. For each feature, it measures how much
that feature reduces prediction error (impurity) across all trees and all splits.

**Advantages:** Fast, no additional computation.
**Limitation:** Can overweight high-cardinality features.

### SHAP (SHapley Additive exPlanations)

Uses `shap.TreeExplainer` on a subsample of 200 data points to compute the marginal
contribution of each feature to each prediction.

**Advantages:** Shows directionality (does high WC increase or decrease predicted ΔP?),
identifies feature interactions.
**Limitation:** Computationally expensive (1–2 minutes).

---

## Beggs & Brill Benchmark

### Implementation

The Beggs & Brill (1973) correlation is implemented from the original paper. It computes
ΔP as the sum of three pressure gradient components integrated over the well depth:

```
ΔP = ∫₀ᴴ (dP/dz)_hydrostatic + (dP/dz)_friction + (dP/dz)_acceleration  dz
```

The implementation includes:
- Standing (1947) correlations for PVT properties (Rs, Bo)
- Papay correlation for gas z-factor
- Beggs & Brill flow regime determination (segregated, intermittent, distributed, transition)
- Horizontal holdup calculation with inclination correction
- Moody friction factor approximation

### Assumed Parameters

The raw Volve CSV does **not** contain well completion data. The following values are assumed
based on representative North Sea well designs:

| Parameter | Assumed Value | Source |
|-----------|:------------:|--------|
| Tubing ID | 4.892 inches | Representative for 6-5/8" casing |
| Well depth (TVD) | 3,100 m | Approximate Volve reservoir depth |
| Inclination | 65° | Typical for deviated wells |
| Oil API gravity | 28° | Medium crude oil |
| Gas specific gravity | 0.65 | — |

> ⚠️ **These assumptions affect only the B&B benchmark, not the RF model.**
> The RF model uses only directly measured columns from the production data.

---

## Assumptions & Simplifications

| # | Assumption | Impact | Mitigable? |
|---|-----------|--------|------------|
| 1 | Well geometry is assumed for B&B | B&B absolute errors may be wrong; RF unaffected | Yes — source real completion data |
| 2 | Steady-state filter thresholds chosen empirically | Some transient data may leak through | Minor — thresholds are conservative |
| 3 | Single field (Volve) only | Transferability to other fields unknown | Yes — test on additional fields |
| 4 | No gas lift or ESP wells | Model doesn't account for artificial lift | Scope limitation |
| 5 | Daily averages used (not sub-daily) | Intra-day transients are smoothed out | Raw data limitation |
| 6 | Temperature is used as input, not modeled | In deployment, T_downhole may not be available | Known limitation |

---

*This document describes the methodology as implemented in `code/vlp_pipeline_v2.py`.
All numbers in the results should be traceable to a single run of that script.*
