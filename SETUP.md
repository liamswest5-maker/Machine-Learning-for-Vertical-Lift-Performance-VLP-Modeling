# 📖 Setup & Execution Guide

Step-by-step instructions for running every component of this project.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Option A: Google Colab (No Installation)](#option-a-google-colab-no-installation)
- [Option B: Run Locally (Python Script)](#option-b-run-locally-python-script)
- [Option C: Run the Jupyter Notebooks Locally](#option-c-run-the-jupyter-notebooks-locally)
- [Option D: Streamlit Web App](#option-d-streamlit-web-app)
- [Option E: Run the Diagnostic Script](#option-e-run-the-diagnostic-script)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum Version | Check Command |
|-------------|:--------------:|---------------|
| Python | 3.8+ | `python --version` |
| pip | 20.0+ | `pip --version` |
| Git | Any | `git --version` |
| Disk space | ~200 MB | — |

> **Note:** If you only want to run the notebooks, use Google Colab (Option A) — no local installation needed.

---

## Option A: Google Colab (No Installation)

This is the **easiest way** to run the project. No Python installation required.

### Step 1: Download the Required Files

From this repository, download:
- `notebooks/01_Data_Analysis.ipynb`
- `notebooks/02_ML_Model.ipynb`
- `data/volve_welldata_raw.csv`

### Step 2: Open Google Colab

Go to [https://colab.research.google.com/](https://colab.research.google.com/)

### Step 3: Run Notebook 1 (Data Analysis)

1. Click **File → Upload notebook**
2. Upload `01_Data_Analysis.ipynb`
3. When the notebook opens, click **Runtime → Run all**
4. When prompted by a file upload cell, upload `volve_welldata_raw.csv`
5. Wait for all cells to complete (takes ~2 minutes)
6. Review the output: data statistics, feature distributions, per-well summaries

### Step 4: Run Notebook 2 (ML Model)

1. Click **File → Upload notebook**
2. Upload `02_ML_Model.ipynb`
3. Click **Runtime → Run all**
4. When prompted, upload `volve_welldata_raw.csv` again
5. Wait for all cells to complete (takes ~5–10 minutes due to model training)
6. Review the output: validation results, figures, feature importance

### Step 5: Download Results

The notebooks will generate figures and CSV files inline. You can also download them from the Colab file browser (folder icon on the left sidebar).

---

## Option B: Run Locally (Python Script)

This runs the **complete pipeline** in one command and produces all results.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling.git
cd Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling
```

### Step 2: Create a Virtual Environment (Recommended)

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, matplotlib, seaborn, shap, streamlit, joblib.

### Step 4: Run the Main Pipeline

```bash
python code/vlp_pipeline_v2.py
```

**What happens:**
- The script runs through 8 stages automatically
- Progress is printed to the terminal at each stage
- Total runtime: approximately **5–15 minutes** (depending on your machine)
- SHAP analysis is the slowest part (~2 minutes)

**What it produces:**

| Output | Location | Description |
|--------|----------|-------------|
| LOWO results | `data/lowo_results.csv` | Per-well cross-validation metrics |
| Chrono results | `data/chrono_split_results.csv` | Chronological split metrics |
| Pooled results | `data/pooled_split_results.csv` | Random split metrics |
| Summary | `data/validation_summary.csv` | All 3 strategies compared |
| Feature importance | `data/feature_importance_gini.csv` | Gini-based rankings |
| SHAP importance | `data/feature_importance_shap.csv` | SHAP-based rankings |
| B&B results | `data/beggs_brill_results.csv` | Beggs & Brill benchmark |
| Comparison | `data/rf_vs_bb_comparison.csv` | RF vs B&B side-by-side |
| Clean dataset | `data/modelready_features.csv` | Feature-engineered dataset |
| 8 figures | `figures/fig1_*.png` through `fig8_*.png` | Publication-quality plots |

### Step 5: View the Results

Open the generated figures in `figures/` and CSVs in `data/` with any viewer or spreadsheet application.

---

## Option C: Run the Jupyter Notebooks Locally

### Step 1: Install Jupyter

```bash
pip install jupyter
```

### Step 2: Launch Jupyter

```bash
jupyter notebook
```

This opens a browser window. Navigate to `notebooks/` and open:
1. `01_Data_Analysis.ipynb` — run all cells (Shift+Enter through each cell)
2. `02_ML_Model.ipynb` — run all cells

> **Note:** The notebooks expect `data/volve_welldata_raw.csv` to exist in the repository root. If running locally, this is already there. If running on Colab, you'll be prompted to upload it.

---

## Option D: Streamlit Web App

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline First (if you haven't already)

The Streamlit app needs the processed data. If you haven't run the pipeline yet:

```bash
python code/vlp_pipeline_v2.py
```

### Step 3: Launch the App

```bash
streamlit run app/streamlit_app.py
```

### Step 4: Use the App

- Your browser will open automatically at `http://localhost:8501`
- If it doesn't, open that URL manually
- Use the sidebar to navigate between different views:
  - **Predictions** — enter well conditions for instant ΔP predictions
  - **Model Performance** — view LOWO and validation results
  - **Feature Importance** — explore what drives predictions
  - **Data Explorer** — browse and filter the dataset

### Step 5: Stop the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

## Option E: Run the Diagnostic Script

This script provides detailed per-well statistics to understand data distribution and why certain wells are harder to predict.

```bash
python code/diagnose_data.py
```

**Output includes:**
- Dataset overview (row counts by well type)
- Per-well statistics for all key variables (min, mean, max, std)
- ΔP range overlap analysis
- Water cut range per well
- Feature correlations with ΔP (whole dataset + per-well)
- Operating regime summary table

---

## Understanding the Output

### Terminal Output

When you run `vlp_pipeline_v2.py`, the terminal shows progress through 8 stages:

```
========================================================================
  STAGE 1/8: Load and filter raw Volve production data
========================================================================
  Raw file: 15634 rows, 27 columns
  Producer rows for usable wells: 7029
  After validity filter: 4176 rows

  ...

========================================================================
  PIPELINE COMPLETE
  Every number above came from this single run.
  Cite these numbers in your report, not earlier drafts.
========================================================================
```

### Key Figures

| Figure | What It Shows |
|--------|--------------|
| `fig1_lowo_predicted_vs_actual.png` | Scatter plot of actual vs predicted ΔP for each held-out well |
| `fig2_chrono_split_timeseries.png` | Time series of actual vs predicted for the last 20% of each well |
| `fig3_feature_importance_gini.png` | Horizontal bar chart of feature importance |
| `fig4_shap_beeswarm.png` | SHAP beeswarm plot showing feature impact direction |
| `fig5_rf_vs_bb.png` | Bar chart comparing RF RMSE vs Beggs & Brill RMSE |
| `fig6_vlp_curve_by_wc.png` | VLP scatter plot colored by water cut |
| `fig7_residual_analysis.png` | 4-panel residual analysis (vs predicted, histogram, vs WC, vs GOR) |
| `fig8_validation_comparison.png` | 3-strategy comparison (LOWO, chrono, pooled) |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"

```bash
pip install scikit-learn
```

### "ModuleNotFoundError: No module named 'shap'"

```bash
pip install shap
```

If SHAP fails to install (common on some systems), the pipeline will still run — it simply skips SHAP analysis and uses Gini importance only.

### "FileNotFoundError: data/volve_welldata_raw.csv"

Make sure you're running the script from the **repository root directory**, not from inside `code/`:

```bash
# ✅ Correct — run from repo root
cd Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling
python code/vlp_pipeline_v2.py

# ❌ Wrong — don't cd into code/
cd code
python vlp_pipeline_v2.py
```

### "UnicodeEncodeError" on Windows

The pipeline handles this automatically. If you still see encoding errors, try:

```bash
set PYTHONIOENCODING=utf-8
python code/vlp_pipeline_v2.py
```

### Streamlit doesn't open in browser

Manually navigate to `http://localhost:8501` in your browser. If that doesn't work, check if another process is using port 8501:

```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Pipeline runs slowly

- **SHAP analysis** takes 1–2 minutes on most machines. This is normal.
- **GridSearchCV** tests 8 hyperparameter combinations with 3-fold CV. ~2 minutes.
- Total expected runtime: **5–15 minutes**.

### Python version issues

This project requires Python 3.8+. Check your version:

```bash
python --version
```

If you have multiple Python versions, try `python3` instead of `python`.

---

## Deploying to Streamlit Cloud (Free Hosting)

Deploy the interactive web app so anyone can access it via a public URL — no installation needed for users.

### Prerequisites

- A **GitHub account** with this repository pushed to it
- A **[Streamlit Cloud](https://share.streamlit.io/)** account (free, sign in with GitHub)
- The pipeline must have been run at least once (`data/rf_model.pkl` and CSVs must exist)

### Step 1: Push to GitHub

Make sure all files are committed and pushed:

```bash
git add -A
git commit -m "Add trained model and deployment config"
git push origin main
```

> **Important:** Make sure `data/rf_model.pkl`, `data/modelready_features.csv`, and all CSV result files are committed. The `.gitignore` is configured to allow these.

### Step 2: Sign in to Streamlit Cloud

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click **Sign in with GitHub**
3. Authorize Streamlit to access your repositories

### Step 3: Deploy the App

1. Click **"New app"**
2. Fill in:
   - **Repository:** Select your `Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling` repo
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
3. Click **"Deploy!"**

### Step 4: Wait for Build

- Streamlit Cloud will install dependencies from `requirements.txt` automatically
- First build takes 3–5 minutes
- You'll get a public URL like: `https://your-app-name.streamlit.app`

### Step 5: Share the Link

Once deployed, share the public URL with anyone. They can use the app directly in their browser — no Python or installation required.

### Updating the Deployed App

Whenever you push changes to GitHub, Streamlit Cloud automatically redeploys:

```bash
git add -A
git commit -m "Update app"
git push origin main
```

The app will update within 1–2 minutes.

### Troubleshooting Deployment

**"ModuleNotFoundError" on Streamlit Cloud:**
- Make sure the module is listed in `requirements.txt`
- Check that version constraints are compatible with Python 3.9+ (Streamlit Cloud default)

**App crashes on load:**
- Check the logs in Streamlit Cloud dashboard (click "Manage app" → "Logs")
- Most common cause: missing data files. Make sure `data/rf_model.pkl` is committed

**Large file issues:**
- If `data/volve_welldata_raw.csv` (1.4 MB) or `rf_model.pkl` exceeds GitHub limits, consider using Git LFS
- For this project, file sizes are well within GitHub's 100 MB limit

---

*For additional questions or issues, refer to the [PROJECT_LOG.md](PROJECT_LOG.md) for known issues and corrections.*

