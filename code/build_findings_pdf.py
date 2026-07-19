import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_LEFT, spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'], spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'], spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='Caveat', parent=styles['Normal'], textColor=colors.HexColor('#B71C1C'), leading=13, spaceAfter=8))
styles.add(ParagraphStyle(name='Caption', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=14))

doc = SimpleDocTemplate('FINDINGS_SUMMARY.pdf', pagesize=letter,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)
story = []

def P(text, style='Justify'):
    story.append(Paragraph(text, styles[style]))

def hr():
    story.append(HRFlowable(width='100%', thickness=0.6, color=colors.HexColor('#999999'), spaceBefore=4, spaceAfter=10))

# ── Title page ──────────────────────────────────────────────────────────
story.append(Spacer(1, 1.2*inch))
story.append(Paragraph('Development of a Random Forest-Based Vertical Lift<br/>'
                        'Performance Model for Multiphase Wellbore Flow Prediction',
                        styles['Title']))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph('Findings Summary — Reproducible Pipeline Results', styles['Heading2']))
story.append(Spacer(1, 0.6*inch))
P('This document summarises the current, verified state of the project: what the data '
  'actually contains, what the model actually achieves, and what still needs to be done '
  'before submission. Every number here was produced by a single script '
  '(<b>code/vlp_pipeline_final.py</b>) included in this package, run once, top to bottom, '
  'against the raw Volve field dataset with no manual editing of intermediate results.')
story.append(PageBreak())

# ── 1. Aim & Objectives ─────────────────────────────────────────────────
P('1. Aim and Objectives', 'H1')
P('<b>Aim:</b> Develop, validate and present a data-driven Vertical Lift Performance (VLP) '
  'model that predicts wellbore pressure drop (Pwf − Pwh) using Random Forest regression '
  'trained on real Volve field production data, benchmarked honestly against a classical '
  'empirical correlation (Beggs &amp; Brill, 1973).')
P('<b>Objectives:</b>')
for obj in [
    'Construct a feature-engineered dataset from the open Volve field dataset using only '
    'directly measured or derived production variables.',
    'Apply a reproducible, threshold-justified steady-state filter to remove transient '
    'production days before training.',
    'Train a Random Forest model and validate it using leave-one-well-out cross-validation '
    'across all usable wells, not a single arbitrary train/test split.',
    'Benchmark against the Beggs &amp; Brill (1973) correlation.',
    'Interpret the model via feature importance to identify which physical variables '
    'dominate pressure-drop prediction.',
]:
    P(f'&bull; {obj}')
hr()

# ── 2. Data ──────────────────────────────────────────────────────────────
P('2. Data Source and Filtering', 'H1')
P('Source: Volve open field dataset (Equinor), well production table. Total 15,634 rows, '
  '9,143 producer (OP) rows, 6,491 water-injector (WI) rows.')
P('<b>Usable wells:</b> Of the 7 wellbores in the file, only 5 have any usable downhole '
  'pressure-gauge data for producer operation: 15/9-F-1C, F-11H, F-12H, F-14H, F-15D. '
  'Two wells sometimes referenced in early drafts of this project — F-4 AH and F-5 AH — '
  'are <b>not usable</b>: F-4 AH is a water injector for its entire production history, and '
  'F-5 AH has zero valid downhole pressure readings across its 144 producer rows.')

data = [
    ['Stage', 'Rows', 'Notes'],
    ['Producer rows (WELL_TYPE = OP)', '9,143', 'Includes rows with no valid pressure/rate data'],
    ['Valid pressure + rate + tubing dP', '5,935', 'Basic completeness filter'],
    ['After steady-state filter', '5,629 (94.9%)', 'CV(q_liq) < 0.35, rolling std(delta_P) < 30 bar over 7-day window'],
]
t = Table(data, colWidths=[2.6*inch, 1.3*inch, 2.6*inch])
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565C0')),
    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTSIZE',(0,0),(-1,-1),9),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
    ('VALIGN',(0,0),(-1,-1),'TOP'),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#F5F5F5')]),
]))
story.append(t)
story.append(Spacer(1,10))
P('Target variable: delta_P = AVG_DOWNHOLE_PRESSURE − AVG_WHP_P (bar). Confirmed to match '
  'the dataset\'s own AVG_DP_TUBING column (correlation 0.9997) — this is a real, measured '
  'quantity, not derived from any assumed physics.')
P('Features used for the core Random Forest model (11 total, all directly measured or '
  'simply derived): normalized oil/gas/water rate, liquid rate, water cut, producing GOR, '
  'wellhead pressure, wellhead temperature, downhole temperature, choke size, and on-stream '
  'hours. No assumed well geometry or PVT correlation is required for the core model.')
hr()

# ── 3. Validation strategy ────────────────────────────────────────────────
P('3. Validation Strategy: Leave-One-Well-Out', 'H1')
P('With only 5 usable wells, a single arbitrary train/test split (e.g. "always test on '
  'F-1C") is fragile and easy to cherry-pick. This project instead uses <b>leave-one-well-out '
  'cross-validation</b>: the model is trained on 4 wells and tested on the 5th, repeated '
  '5 times so every well is held out exactly once. All 5 results are reported — not the '
  'single best-looking one.')

lowo = pd.read_csv('data/lowo_results.csv')
tbl_data = [['Held-out well','n_test','n_train','RMSE (bar)','MAE (bar)','MAPE (%)','R2']]
for _,r in lowo.iterrows():
    tbl_data.append([r['test_well'], int(r['n_test']), int(r['n_train']),
                      f"{r['RMSE']:.2f}", f"{r['MAE']:.2f}", f"{r['MAPE']:.2f}", f"{r['R2']:.3f}"])
mean_row = ['Mean across folds','','', f"{lowo.RMSE.mean():.2f}", f"{lowo.MAE.mean():.2f}",
            f"{lowo.MAPE.mean():.2f}", f"{lowo.R2.mean():.3f}"]
std_row  = ['Std across folds','','', f"{lowo.RMSE.std():.2f}", f"{lowo.MAE.std():.2f}",
            f"{lowo.MAPE.std():.2f}", f"{lowo.R2.std():.3f}"]
tbl_data.append(mean_row); tbl_data.append(std_row)

t2 = Table(tbl_data, colWidths=[1.3*inch,0.6*inch,0.6*inch,0.85*inch,0.85*inch,0.75*inch,0.6*inch])
t2.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1565C0')),
    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTSIZE',(0,0),(-1,-1),8.5),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
    ('ROWBACKGROUNDS',(0,1),(-1,-3),[colors.white, colors.HexColor('#F5F5F5')]),
    ('BACKGROUND',(0,-2),(-1,-1),colors.HexColor('#FFF3E0')),
    ('FONTNAME',(0,-2),(-1,-1),'Helvetica-Bold'),
]))
story.append(t2)
story.append(Spacer(1,10))
P('<b>Honest reading:</b> mean R2 across folds is near zero, not the 0.88 headline figure '
  'from an earlier single-split result. Performance ranges from R2 = 0.82 (F-11H) to '
  'R2 = -2.37 (F-15D). This variance is itself a legitimate finding: the model generalises '
  'well to some wells and poorly to others, and F-15D — a smaller, lower-rate well — is '
  'where it struggles most. This should be discussed as a limitation, not hidden.')
story.append(Image('figures/fig1_lowo_predicted_vs_actual.png', width=6.6*inch, height=2.85*inch))
P('Figure 1: Predicted vs actual delta_P across all 5 leave-one-well-out folds (left), and '
  'per-fold R2/RMSE (right).', 'Caption')
hr()

# ── 4. Feature importance ─────────────────────────────────────────────────
P('4. Feature Importance', 'H1')
imp = pd.read_csv('data/feature_importance.csv', index_col=0)
imp_data = [['Feature','Importance']] + [[idx, f"{row.iloc[0]:.3f}"] for idx,row in imp.iterrows()]
t3 = Table(imp_data, colWidths=[3*inch, 1.5*inch])
t3.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E7D32')),
    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTSIZE',(0,0),(-1,-1),9),
    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#F5F5F5')]),
]))
story.append(t3)
story.append(Spacer(1,10))
P('Water cut and water rate together account for over half of total feature importance. '
  'This independently supports the project\'s central engineering argument: classical VLP '
  'correlations, developed decades ago on lower water-cut experimental data, are least '
  'reliable exactly where the model finds the most signal — high water cut conditions.')
story.append(Image('figures/fig2_feature_importance.png', width=5.5*inch, height=3.8*inch))
P('Figure 2: Random Forest feature importance (Gini-based), model trained on all 5 wells.', 'Caption')
hr()

# ── 5. Beggs & Brill benchmark ────────────────────────────────────────────
P('5. Beggs and Brill Benchmark — Important Caveat', 'H1')
P('A Beggs &amp; Brill (1973) implementation was run for comparison. The Volve dataset does '
  '<b>not</b> include tubing diameter, well depth, or deviation survey data required by this '
  'correlation.', 'Justify')
story.append(Paragraph(
    'The values used (tubing ID 4.892 in, depth 3,100 m, inclination 65 degrees, API 28, '
    'gas SG 0.65) are <b>assumed, representative placeholders — not measured values from '
    'these wells.</b> The resulting comparison (RF RMSE ~15 bar vs Beggs &amp; Brill RMSE ~126 '
    'bar in an earlier full-dataset run) is directionally consistent with the project\'s '
    'thesis, but should not be quoted as a precise, defensible number until real completion '
    'data replaces these assumptions.', styles['Caveat']))
story.append(Image('figures/fig3_rf_vs_bb.png', width=6.2*inch, height=3.6*inch))
P('Figure 3: RF vs Beggs & Brill RMSE by held-out well. Beggs & Brill uses assumed geometry — treat as illustrative only.', 'Caption')
hr()

# ── 6. What needs to happen before submission ─────────────────────────────
P('6. Outstanding Work Before Submission', 'H1')
for item in [
    'Obtain real tubing ID, depth (MD/TVD) and deviation survey data for the 5 wells from '
    'Volve\'s public completion reports, and re-run the Beggs &amp; Brill comparison with '
    'real values.',
    'Resolve the missing second, independent test dataset. An earlier draft of this project '
    'used a fabricated "Mukherjee &amp; Brill (1985)" dataset that was not sourced from the '
    'actual paper — this has been removed. Either source real published experimental data '
    'or explicitly scope this project to single-field (Volve-only) validation and state '
    'that as a limitation.',
    'Write the methodology and results chapters directly from the numbers in this package, '
    'not from earlier draft narrative (see PROJECT_LOG.md for a full list of discrepancies '
    'between earlier claims and what this reproducible pipeline actually produces).',
    'Consider a within-well time-based split (early vs late life) as a second, more targeted '
    'test of the water-cut degradation thesis, since it isolates the effect the project is '
    'actually arguing for.',
]:
    P(f'&bull; {item}')

doc.build(story)
print('PDF built.')
