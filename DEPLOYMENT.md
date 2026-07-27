# 🚀 Streamlit Deployment Guide

How to deploy the VLP Predictor Streamlit app — both locally and to the internet
via **Streamlit Community Cloud** (free, shareable via URL).

---

## Option 1 — Run Locally (quickest)

### Prerequisites
Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Run the app

From the **repository root** directory:

```bash
streamlit run app/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Model loading:** The app first tries to load `data/rf_model.pkl`. If it does not
> exist, it automatically retrains the model from `data/modelready_features.csv`.
> First-time loading of the pkl (~146 MB) takes a few seconds.

---

## Option 2 — Deploy to Streamlit Community Cloud (free, public URL)

This gives you a permanent shareable URL like
`https://your-app-name.streamlit.app` — ideal for a final-year project presentation.

### Step 1 — Make sure the repo is on GitHub

The project is already a GitHub repository ✅
Repository: `Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling`

### Step 2 — Solve the model file size issue

> ⚠️ **Critical:** GitHub has a **100 MB file size limit**.
> The trained model `data/rf_model.pkl` is **~146 MB** and **cannot be pushed to GitHub**.

**Solution — use the built-in fallback (recommended):**

The app already has a `train_model_from_data()` function that retrains the Random Forest
from `data/modelready_features.csv` (1.2 MB — well within GitHub limits) if no pkl is found.

1. Add `data/rf_model.pkl` to `.gitignore`:

```
# Large trained model — too big for GitHub; app retrains on first start
data/rf_model.pkl
```

2. Make sure `data/modelready_features.csv` **is** committed:

```bash
git add data/modelready_features.csv
git add data/classification_metrics.csv
git commit -m "Add CSVs for cloud deployment"
git push
```

> **On first cloud startup**, the app will say "Training from data..." and take
> ~1-2 minutes. After that, the trained model is cached for the duration of the session.

---

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.

2. Click **"New app"** (top right).

3. Fill in the deployment form:

   | Field | Value |
   |-------|-------|
   | Repository | `your-github-username/Machine-Learning-for-Vertical-Lift-Performance-VLP-Modeling` |
   | Branch | `main` (or `master`) |
   | Main file path | `app/streamlit_app.py` |
   | App URL (optional) | e.g. `vlp-predictor` gives `vlp-predictor.streamlit.app` |

4. Click **"Deploy!"**

5. Streamlit Cloud will:
   - Clone your repo
   - Install packages from `requirements.txt` automatically
   - Start the app (retraining the model on first run)

6. First startup takes **2-3 minutes** (package install + model retrain).
   After that, the app is live at your chosen URL.

---

### Step 4 — Share your app

The app is live at:

```
https://machine-learning-for-vertical-lift-performance-vlp-modeling-7k.streamlit.app/
```

Share this URL in your project report, viva presentation, or with collaborators.


## Configuration File

The repo already has `.streamlit/config.toml`. Make sure it contains:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named seaborn` | Already in requirements.txt — reinstall |
| App fails on pkl load | Delete rf_model.pkl from repo — app will retrain automatically |
| "File too large" on git push | Add data/rf_model.pkl to .gitignore |
| Slow first load on cloud | Expected — model retrains once per cold start (~1-2 min) |
| `data/classification_metrics.csv` not found | Run `python code/vlp_pipeline_v2.py` locally, commit the CSV |

---

## What to Commit vs. Exclude

| File | Include? | Reason |
|------|:--------:|--------|
| `app/streamlit_app.py` | YES | Main app |
| `requirements.txt` | YES | Cloud installs these |
| `.streamlit/config.toml` | YES | App configuration |
| `data/modelready_features.csv` | YES | Needed for retraining |
| `data/lowo_results.csv` | YES | Pre-computed metrics |
| `data/chrono_split_results.csv` | YES | Pre-computed metrics |
| `data/pooled_split_results.csv` | YES | Pre-computed metrics |
| `data/validation_summary.csv` | YES | Dashboard summary |
| `data/classification_metrics.csv` | YES | Accuracy/Precision/Recall/F1 |
| `data/feature_importance_gini.csv` | YES | Feature importance tab |
| `data/rf_vs_bb_comparison.csv` | YES | Beggs & Brill benchmark tab |
| `data/rf_model.pkl` | **NO** | 146 MB — too large for GitHub |

*After deployment, share your URL in your project report and viva presentation.*
