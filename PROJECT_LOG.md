# Project Log — Honest Record of This Working Session

## Project title
Development of a Random Forest-Based Vertical Lift Performance Model for
Multiphase Wellbore Flow Prediction.

## Aim
Develop, validate and present a data-driven VLP model that predicts wellbore
pressure drop (Pwf − Pwh) using Random Forest regression trained on real
Volve field production data, and honestly compare it against a classical
correlation (Beggs & Brill).

## What happened earlier in this project's chat history (before this package)

An earlier working session generated a large amount of code and narrative
that sounded confident and detailed but contained real problems. This log
exists so you know exactly what to trust and what to discard.

### Confirmed correct
- Volve dataset structure: 15,634 rows, 9,143 producer (OP) rows, 6,491
  injector (WI) rows.
- The five wells with any usable downhole pressure gauge data: F-1C, F-11H,
  F-12H, F-14H, F-15D.
- `AVG_DP_TUBING` in the raw file is essentially the same as
  `AVG_DOWNHOLE_PRESSURE − AVG_WHP_P` (correlation 0.9997) — it is a real
  measured tubing pressure drop, not something invented.

### Confirmed WRONG or fabricated — do not repeat these in your report
1. **Well selection narrative.** Earlier text described F-4 AH and F-5 AH as
   good "PDG wells" with specific behavioral claims ("F-4 shows liquid
   loading late in life," "F-5 is a high-GOR well"). Checked directly:
   F-4 AH is **100% a water injector** (never produced) and F-5 AH's 144
   producer rows have **zero valid downhole pressure readings**. Neither
   well can be used for VLP at all. The actual code that ran later used the
   correct 5 wells — only the narrative describing F-4/F-5 was fabricated.
2. **Inconsistent row counts across the same claimed filter.** One message
   reported 4,189 rows surviving a steady-state filter; a later message
   reported 3,282 rows from "removing 2,669 of 5,951" using ostensibly the
   same method. These cannot both be true of one filter run. Neither number
   should be used — use the counts in `lowo_results.csv` from this package.
3. **A hardcoded "Mukherjee & Brill (1985)" dataset was fabricated.** Code
   presented 20 tuples as "Extracted from SPE-12372 Table 2" with no
   retrieval or citation step anywhere in that conversation. This is
   invented data dressed up as a literature source and must not be cited
   as such. It has been excluded entirely from this package.
4. **A claimed physics feature pipeline (tubing ID, well depth, deviation
   survey "from Volve well completion reports") was never actually loaded**
   — the raw CSV does not contain those columns, and no completion report
   file was ever fetched or read in that conversation. Any dimensionless
   groups (N_Lv, N_gv, Re, Fr) computed "from" that geometry in earlier
   code were built on invented constants presented as if measured.
5. **The single headline result (RMSE=5.51 bar, MAPE=2.47%, R²=0.88)** came
   from one arbitrary train/test split (train on 4 wells, test on F-1C
   only). It was never checked against the other 4 possible well-holdout
   splits. When checked (this package), R² across all 5 folds ranges from
   **−2.37 to +0.82**, with a mean near 0. The single number was real but
   badly unrepresentative — not fabricated, but presented without the
   variance that would have revealed how fragile it was.
6. **My own error, corrected in this package:** an early quick recheck of
   the steady-state filter (done in this same session, before the final
   script) sorted rows by date as raw text instead of parsing dates
   properly. The raw CSV mixes two date formats (`4/7/2014` and
   `24-Jul-13`), which silently breaks string-sorting and corrupts any
   rolling-window calculation. That mistake produced a wrong intermediate
   number (52.7% retained). After parsing dates correctly, the real
   figure is 94.9% retained. This is recorded here so the discrepancy
   between numbers you may have already written down and the final
   package is understood, not mistaken for yet another fabrication.

## What is solid and reproducible right now
- `code/vlp_pipeline_final.py` runs start to finish on the raw CSV with no
  manual steps, using only real measured columns for the core model.
- Leave-one-well-out cross-validation across all 5 real wells (not one
  arbitrary split) — see `data/lowo_results.csv`.
- Feature importance genuinely computed from the trained model (Gini
  importance, not invented SHAP numbers) — water cut and water rate
  dominate, which independently supports the "classical correlations
  degrade at high water cut" thesis. See `data/feature_importance.csv`.
- The Beggs & Brill comparison in `data/beggs_brill_comparison_ASSUMED_GEOMETRY.csv`
  is real code, correctly implementing the 1973 correlation, but run on
  assumed (not measured) tubing geometry — its absolute numbers should be
  treated as illustrative until real completion data is substituted in.

## Recommended immediate next steps
1. Source real tubing ID / depth / deviation data for the 5 wells (Volve
   public completion reports) and re-run the Beggs & Brill comparison.
2. Decide how to handle the missing second, independent test dataset —
   either source real published data or explicitly scope the project to
   single-field validation with that stated as a limitation.
3. Write the methodology and results chapters directly from this package's
   numbers, not from earlier draft narrative.
