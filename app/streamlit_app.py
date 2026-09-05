"""
Hybrid VLP Prediction - Streamlit Web Application
=================================================
Physics-informed hybrid Random Forest:
  ΔP_hybrid = ΔP_baseline + RF(residual)

Author: Williams Iwum
Institution: Federal University of Petroleum Resources, Effurun
Dataset: Volve Field (Equinor)

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Hybrid VLP Predictor - Physics-Informed RF",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS (same style as your original app)
# =====================================================================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1565C0;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B2838 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E0E0E0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# PATHS & PHYSICS CONSTANTS (dissertation Table 3.2)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

HYBRID_DATA = os.path.join(DATA_DIR, "hybrid_model_ready.csv")
FALLBACK_DATA = os.path.join(DATA_DIR, "modelready_features.csv")
POOLED_PATH = os.path.join(DATA_DIR, "hybrid_pooled_results.csv")
LOWO_PATH = os.path.join(DATA_DIR, "hybrid_lowo_results.csv")
CHRONO_PATH = os.path.join(DATA_DIR, "hybrid_chronological_results.csv")
PER_WELL_PATH = os.path.join(DATA_DIR, "hybrid_pooled_per_well.csv")
FI_PATH = os.path.join(DATA_DIR, "hybrid_feature_importance.csv")

D, TVD = 0.0889, 2750.0
RHO_OIL, RHO_WAT, GAS_SG = 850.0, 1030.0, 0.75
MU_MIX, EPS, G = 0.003, 0.0457e-3, 9.81
P_STD, T_STD, R_AIR = 101325.0, 288.15, 287.058

PLAIN_FEATURES = [
    "q_oil", "q_gas", "q_wat", "q_liq", "WC", "GOR", "GLR",
    "ln_qliq", "ln_qoil", "ln_qgas", "dT", "choke_pct", "ON_STREAM_HRS",
]
HYBRID_FEATURES = PLAIN_FEATURES + ["delta_P_baseline"]

RF_PARAMS = dict(
    n_estimators=400,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=42,
    n_jobs=-1,
)


# =====================================================================
# PHYSICS BASELINE
# =====================================================================
def swamee_jain(Re, rel_rough):
    Re = max(float(Re), 1.0)
    term = rel_rough / 3.7 + 5.74 / (Re ** 0.9)
    return float(np.clip(0.25 / (np.log10(term) ** 2), 0.008, 0.1))


def compute_baseline_single(q_oil, q_gas, q_wat, P_wf, P_wh, T_wf, T_wh):
    """Homogeneous no-slip ΔP_baseline [bar]."""
    q_oil_s = q_oil / 86400.0
    q_wat_s = q_wat / 86400.0
    q_gas_s = q_gas / 86400.0
    P_avg = 0.5 * (P_wf + P_wh)
    T_avg_K = 0.5 * (T_wf + T_wh) + 273.15
    P_avg_Pa = P_avg * 1e5
    rho_g = (P_avg_Pa * GAS_SG) / (R_AIR * max(T_avg_K, 1.0))
    q_g_dh = q_gas_s * (P_STD / max(P_avg_Pa, 1.0)) * (T_avg_K / T_STD)
    q_liq_s = q_oil_s + q_wat_s
    q_tot = q_liq_s + q_g_dh
    f_liq = q_liq_s / q_tot if q_tot > 0 else 1.0
    rho_liq = (
        (q_oil_s * RHO_OIL + q_wat_s * RHO_WAT) / max(q_liq_s, 1e-12)
        if q_liq_s > 0
        else RHO_OIL
    )
    rho_mix = f_liq * rho_liq + (1.0 - f_liq) * rho_g
    A = np.pi / 4.0 * D ** 2
    v_mix = q_tot / A
    Re = (rho_mix * v_mix * D) / MU_MIX
    f = swamee_jain(Re, EPS / D)
    dP_hyd = (rho_mix * G * TVD) / 1e5
    dP_fric = (f * (TVD / D) * (rho_mix * v_mix ** 2 / 2.0)) / 1e5
    return dP_hyd + dP_fric


def build_feature_row(q_oil, q_gas, q_wat, P_wf, P_wh, T_wf, T_wh, choke, on_hrs):
    q_liq = q_oil + q_wat
    WC = q_wat / q_liq if q_liq > 0 else 0.0
    GOR = q_gas / max(q_oil, 1e-6)
    GLR = q_gas / max(q_liq, 1e-6)
    dT = T_wf - T_wh
    baseline = compute_baseline_single(q_oil, q_gas, q_wat, P_wf, P_wh, T_wf, T_wh)
    row = {
        "q_oil": q_oil,
        "q_gas": q_gas,
        "q_wat": q_wat,
        "q_liq": q_liq,
        "WC": WC,
        "GOR": GOR,
        "GLR": GLR,
        "ln_qliq": np.log(max(q_liq, 1e-6)),
        "ln_qoil": np.log(max(q_oil, 1e-6)),
        "ln_qgas": np.log(max(q_gas, 1e-6)),
        "dT": dT,
        "choke_pct": choke,
        "ON_STREAM_HRS": on_hrs,
        "delta_P_baseline": baseline,
    }
    return row, baseline


# =====================================================================
# LOAD / TRAIN MODELS
# =====================================================================
@st.cache_resource
def get_models():
    path = HYBRID_DATA if os.path.exists(HYBRID_DATA) else FALLBACK_DATA
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    # Harmonise column names if old CSV is used
    if "delta_P" in df.columns and "delta_P_measured" not in df.columns:
        df["delta_P_measured"] = df["delta_P"]
    if "AVG_CHOKE_SIZE_P" in df.columns and "choke_pct" not in df.columns:
        df["choke_pct"] = df["AVG_CHOKE_SIZE_P"]
    if "log_q_liq" in df.columns and "ln_qliq" not in df.columns:
        df["ln_qliq"] = df["log_q_liq"]
        df["ln_qoil"] = df.get("log_q_oil", np.log(np.maximum(df["q_oil"], 1e-6)))
        df["ln_qgas"] = df.get("log_q_gas", np.log(np.maximum(df["q_gas"], 1e-6)))

    # Recompute baseline if missing
    if "delta_P_baseline" not in df.columns:
        baselines = []
        for _, r in df.iterrows():
            baselines.append(
                compute_baseline_single(
                    r["q_oil"], r["q_gas"], r["q_wat"],
                    r.get("AVG_DOWNHOLE_PRESSURE", r.get("P_wf", 200)),
                    r["AVG_WHP_P"],
                    r["AVG_DOWNHOLE_TEMPERATURE"],
                    r["AVG_WHT_P"],
                )
            )
        df["delta_P_baseline"] = baselines
    if "residual" not in df.columns:
        df["residual"] = df["delta_P_measured"] - df["delta_P_baseline"]

    need = [c for c in PLAIN_FEATURES + ["delta_P_baseline", "residual", "delta_P_measured"] if c in df.columns]
    df = df.dropna(subset=need)

    plain = RandomForestRegressor(**RF_PARAMS)
    plain.fit(df[PLAIN_FEATURES], df["delta_P_measured"])
    hybrid = RandomForestRegressor(**RF_PARAMS)
    hybrid.fit(df[HYBRID_FEATURES], df["residual"])

    return {
        "plain": plain,
        "hybrid": hybrid,
        "df": df,
        "n_samples": len(df),
        "mean_dp": float(df["delta_P_measured"].mean()),
        "std_dp": float(df["delta_P_measured"].std()),
    }


def predict_with_uncertainty(model, X):
    """Mean and std across trees."""
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    return float(tree_preds.mean()), float(tree_preds.std())


# =====================================================================
# SIDEBAR (your name + project branding)
# =====================================================================
with st.sidebar:
    st.markdown("# 🛢️ Hybrid VLP Predictor")
    st.markdown("---")
    st.markdown(
        """
**Project:** Physics-Informed Hybrid Random Forest  
for Vertical Lift Performance Prediction

**Author:** Williams Iwum

**Institution:** Federal University of Petroleum Resources, Effurun

**Programme:** MEng, Petroleum Engineering

**Dataset:** Volve Field (Equinor)
"""
    )
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to:",
        [
            "🏠 Dashboard",
            "📊 Data Analytics",
            "🔮 Single Prediction",
            "📈 VLP Curve Generator",
            "📉 Model Performance",
            "🔍 Feature Importance",
            "📁 Batch Prediction",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

# =====================================================================
# LOAD
# =====================================================================
models = get_models()
if models is None:
    st.error("No training data found.")
    st.info(
        f"Expected `{HYBRID_DATA}` (run `python code/hybrid_rf_pipeline.py`) "
        f"or `{FALLBACK_DATA}`."
    )
    st.stop()

plain_model = models["plain"]
hybrid_model = models["hybrid"]
analytics_df = models["df"]

# =====================================================================
# DASHBOARD
# =====================================================================
if page == "🏠 Dashboard":
    st.markdown(
        '<p class="main-header">Physics-Informed Hybrid VLP Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">ΔP_hybrid = ΔP_baseline + RF(residual) — Volve Field Case Study</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", f"{models['n_samples']:,}")
    c2.metric("Features (plain / hybrid)", f"{len(PLAIN_FEATURES)} / {len(HYBRID_FEATURES)}")
    c3.metric("Mean measured ΔP", f"{models['mean_dp']:.1f} bar")
    c4.metric("Std ΔP", f"{models['std_dp']:.1f} bar")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("### Validation snapshot")
        if os.path.exists(POOLED_PATH):
            st.caption("Pooled random 80/20")
            st.dataframe(pd.read_csv(POOLED_PATH).round(3), use_container_width=True, hide_index=True)
        if os.path.exists(CHRONO_PATH):
            st.caption("Chronological 70/30")
            st.dataframe(pd.read_csv(CHRONO_PATH).round(3), use_container_width=True, hide_index=True)
        if not os.path.exists(POOLED_PATH):
            st.info("Run `hybrid_rf_pipeline.py` to populate validation CSVs.")

    with right:
        st.markdown("### Hybrid feature importance (top)")
        if os.path.exists(FI_PATH):
            fi = pd.read_csv(FI_PATH)
            hy = fi[fi["model"] == "Hybrid RF"].sort_values("importance", ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(hy["feature"], hy["importance"], color="#2E7D32", alpha=0.85)
            ax.set_xlabel("Importance")
            ax.set_title("Hybrid RF — top features")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature importance CSV not found yet.")

    st.markdown("---")
    st.markdown(
        """
### About this app
This application implements the **two-stage hybrid model** from the MEng dissertation:

1. **Stage 1 — Mechanistic baseline:** homogeneous, no-slip pressure drop (hydrostatic + Swamee–Jain friction)  
2. **Stage 2 — Random Forest residual:** learns only the correction \(r = \Delta P_{\mathrm{measured}} - \Delta P_{\mathrm{baseline}}\)  
3. **Final prediction:** \(\Delta P_{\mathrm{hybrid}} = \Delta P_{\mathrm{baseline}} + \mathrm{RF}(r)\)

Use the sidebar for single predictions, validation tables, feature importance, and data exploration.
"""
    )

# =====================================================================
# SINGLE PREDICTION
# =====================================================================
elif page == "🔮 Single Prediction":
    st.markdown("## 🔮 Single Well Prediction (Hybrid)")
    st.markdown(
        "Enter operating parameters. The app computes **baseline**, **plain RF**, and **hybrid** side by side."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Flow rates (Sm³/d)")
        q_oil = st.number_input("Oil rate (q_oil)", 0.0, 20000.0, 106.0, 50.0)
        q_gas = st.number_input("Gas rate (q_gas)", 0.0, 1_000_000.0, 16147.0, 500.0)
        q_wat = st.number_input("Water rate (q_wat)", 0.0, 20000.0, 1645.0, 50.0)
    with col2:
        st.markdown("#### Pressures & temperatures")
        P_wf = st.number_input("Downhole pressure Pwf (bar)", 0.0, 400.0, 266.729, 5.0)
        P_wh = st.number_input("Wellhead pressure Pwh (bar)", 0.0, 300.0, 26.846, 5.0)
        T_wf = st.number_input("Downhole temperature (°C)", 0.0, 250.0, 107.903, 5.0)
        T_wh = st.number_input("Wellhead temperature (°C)", 0.0, 200.0, 79.765, 5.0)
    with col3:
        st.markdown("#### Operations")
        choke = st.number_input("Choke size (%)", 0.0, 100.0, 59.0, 5.0)
        on_hrs = st.number_input("On-stream hours", 0.1, 24.0, 24.0, 1.0)

    q_liq = q_oil + q_wat
    wc = q_wat / q_liq if q_liq > 0 else 0.0
    gor = q_gas / q_oil if q_oil > 0 else 0.0
    a, b, c = st.columns(3)
    a.metric("Liquid rate", f"{q_liq:.0f} Sm³/d")
    b.metric("Water cut", f"{wc:.1%}")
    c.metric("GOR", f"{gor:.0f} Sm³/Sm³")

    if st.button("🔮 Predict ΔP", type="primary", use_container_width=True):
        row, baseline = build_feature_row(
            q_oil, q_gas, q_wat, P_wf, P_wh, T_wf, T_wh, choke, on_hrs
        )
        Xp = pd.DataFrame([row])[PLAIN_FEATURES]
        Xh = pd.DataFrame([row])[HYBRID_FEATURES]

        pred_plain, std_plain = predict_with_uncertainty(plain_model, Xp)
        pred_resid, std_resid = predict_with_uncertainty(hybrid_model, Xh)
        pred_hybrid = baseline + pred_resid

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ΔP_baseline", f"{baseline:.2f} bar")
        m2.metric("RF residual", f"{pred_resid:.2f} bar", help=f"Tree std ≈ {std_resid:.2f} bar")
        m3.metric("ΔP_hybrid", f"{pred_hybrid:.2f} bar")
        m4.metric("Plain RF", f"{pred_plain:.2f} bar", help=f"Tree std ≈ {std_plain:.2f} bar")

        st.success(
            f"**Hybrid prediction:** \(P_{{wf}} \\approx P_{{wh}} + \\Delta P_{{\\mathrm{{hybrid}}}} "
            f"= {P_wh:.1f} + {pred_hybrid:.1f} = \\mathbf{{{P_wh + pred_hybrid:.1f}\\,bar}}\)"
        )
        st.caption(
            "Hybrid = mechanistic baseline + learned residual (dissertation Stage 1 + Stage 2)."
        )

# =====================================================================
# VLP CURVE
# =====================================================================
elif page == "📈 VLP Curve Generator":
    st.markdown("## 📈 VLP Curve Generator (Hybrid)")
    st.markdown("Curves use the **hybrid** model at selected water cuts.")

    col1, col2 = st.columns(2)
    with col1:
        base_q_oil = st.number_input("Base oil rate (Sm³/d)", value=1000.0, step=100.0)
        base_q_gas = st.number_input("Base gas rate (Sm³/d)", value=150000.0, step=5000.0)
        P_wf = st.number_input("Pwf (bar)", value=250.0, step=5.0, key="vlp_pwf")
        P_wh = st.number_input("Pwh (bar)", value=50.0, step=5.0, key="vlp_pwh")
    with col2:
        T_wf = st.number_input("Twf (°C)", value=106.0, step=5.0, key="vlp_twf")
        T_wh = st.number_input("Twh (°C)", value=70.0, step=5.0, key="vlp_twh")
        choke = st.number_input("Choke (%)", value=50.0, step=5.0, key="vlp_choke")
        q_min = st.number_input("Min liquid rate", value=100.0, step=50.0)
        q_max = st.number_input("Max liquid rate", value=8000.0, step=500.0)

    wc_scenarios = st.multiselect(
        "Water cut scenarios",
        options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=[0.1, 0.3, 0.5, 0.7],
    )

    if st.button("📈 Generate VLP Curves", type="primary", use_container_width=True):
        q_range = np.linspace(q_min, q_max, 40)
        fig, ax = plt.subplots(figsize=(11, 7))
        colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, max(len(wc_scenarios), 1)))
        gor0 = base_q_gas / max(base_q_oil, 1.0)

        for wc_val, color in zip(wc_scenarios, colors):
            means, stds = [], []
            for q in q_range:
                q_oil_i = q * (1 - wc_val)
                q_wat_i = q * wc_val
                q_gas_i = q_oil_i * gor0
                row, _ = build_feature_row(
                    q_oil_i, q_gas_i, q_wat_i, P_wf, P_wh, T_wf, T_wh, choke, 24.0
                )
                Xh = pd.DataFrame([row])[HYBRID_FEATURES]
                pred_resid, std_r = predict_with_uncertainty(hybrid_model, Xh)
                pred = row["delta_P_baseline"] + pred_resid
                means.append(pred)
                stds.append(std_r)
            means, stds = np.array(means), np.array(stds)
            ax.plot(q_range, means, "-", color=color, lw=2.5, label=f"WC = {wc_val:.0%}")
            ax.fill_between(q_range, means - stds, means + stds, color=color, alpha=0.18)

        ax.set_xlabel("Liquid flow rate (Sm³/d)")
        ax.set_ylabel("ΔP hybrid (bar)")
        ax.set_title("Hybrid VLP curves (±1 tree-std band)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =====================================================================
# MODEL PERFORMANCE
# =====================================================================
elif page == "📉 Model Performance":
    st.markdown("## 📉 Model Performance & Validation")
    st.markdown("Baseline vs **Plain RF** vs **Hybrid RF** under three strategies.")

    tabs = st.tabs(["Pooled", "Leave-One-Well-Out", "Chronological", "Per-well pooled"])

    with tabs[0]:
        if os.path.exists(POOLED_PATH):
            st.dataframe(pd.read_csv(POOLED_PATH).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Missing hybrid_pooled_results.csv — run the pipeline.")

    with tabs[1]:
        if os.path.exists(LOWO_PATH):
            lowo = pd.read_csv(LOWO_PATH)
            st.dataframe(lowo.round(3), use_container_width=True, hide_index=True)
            hy = lowo[lowo["model"] == "Hybrid RF"]
            if len(hy):
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(hy["test_well"].astype(str), hy["R2"], color="#1565C0", alpha=0.85)
                ax.axhline(0, color="gray", ls="--", lw=0.8)
                ax.set_ylabel("R²")
                ax.set_title("LOWO R² — Hybrid RF by held-out well")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Missing hybrid_lowo_results.csv")

    with tabs[2]:
        if os.path.exists(CHRONO_PATH):
            st.dataframe(pd.read_csv(CHRONO_PATH).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Missing hybrid_chronological_results.csv")

    with tabs[3]:
        if os.path.exists(PER_WELL_PATH):
            st.dataframe(pd.read_csv(PER_WELL_PATH).round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Missing hybrid_pooled_per_well.csv")

# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================
elif page == "🔍 Feature Importance":
    st.markdown("## 🔍 Feature Importance — Plain RF vs Hybrid RF")

    if os.path.exists(FI_PATH):
        fi = pd.read_csv(FI_PATH)
        col1, col2 = st.columns(2)
        for col, model_name, color in [
            (col1, "Plain RF", "#1565C0"),
            (col2, "Hybrid RF", "#2E7D32"),
        ]:
            with col:
                st.markdown(f"### {model_name}")
                sub = fi[fi["model"] == model_name].sort_values("importance", ascending=True)
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.barh(sub["feature"], sub["importance"], color=color, alpha=0.85)
                ax.set_xlabel("Gini importance")
                ax.set_title(model_name)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(sub.sort_values("importance", ascending=False).round(4), hide_index=True)

        st.markdown(
            """
### Interpretation (dissertation §4.7)
- **Plain RF** tends to rely on GLR / water cut / water rate (multiphase composition).  
- **Hybrid RF** often ranks **ΔP_baseline** and liquid-rate terms highest — it learns a *correction* to physics, not the full ΔP from scratch.
"""
        )
    else:
        st.info("Run the hybrid pipeline to create hybrid_feature_importance.csv")

# =====================================================================
# BATCH
# =====================================================================
elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Prediction (Hybrid)")
    st.markdown(
        """
Required columns: `q_oil`, `q_gas`, `q_wat`, `AVG_DOWNHOLE_PRESSURE` (or `P_wf`),
`AVG_WHP_P`, `AVG_DOWNHOLE_TEMPERATURE`, `AVG_WHT_P`, `AVG_CHOKE_SIZE_P` (or `choke_pct`), `ON_STREAM_HRS`
"""
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.dataframe(batch.head(10), use_container_width=True, hide_index=True)
        if st.button("🚀 Run batch hybrid prediction", type="primary"):
            preds = []
            for _, r in batch.iterrows():
                q_oil = float(r.get("q_oil", 0))
                q_gas = float(r.get("q_gas", 0))
                q_wat = float(r.get("q_wat", 0))
                P_wf = float(r.get("AVG_DOWNHOLE_PRESSURE", r.get("P_wf", 200)))
                P_wh = float(r.get("AVG_WHP_P", 50))
                T_wf = float(r.get("AVG_DOWNHOLE_TEMPERATURE", 100))
                T_wh = float(r.get("AVG_WHT_P", 70))
                choke = float(r.get("AVG_CHOKE_SIZE_P", r.get("choke_pct", 50)))
                on_hrs = float(r.get("ON_STREAM_HRS", 24))
                row, baseline = build_feature_row(
                    q_oil, q_gas, q_wat, P_wf, P_wh, T_wf, T_wh, choke, on_hrs
                )
                Xh = pd.DataFrame([row])[HYBRID_FEATURES]
                resid = float(hybrid_model.predict(Xh)[0])
                preds.append(baseline + resid)
            batch["Predicted_dP_hybrid_bar"] = preds
            st.dataframe(batch, use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download results",
                batch.to_csv(index=False),
                "hybrid_vlp_predictions.csv",
                "text/csv",
            )

# =====================================================================
# DATA ANALYTICS (kept similar to your original)
# =====================================================================
elif page == "📊 Data Analytics":
    st.markdown("## 📊 Data Analytics & Exploration")
    df = analytics_df.copy()
    if "Date of Production" in df.columns:
        df["Date of Production"] = pd.to_datetime(df["Date of Production"], errors="coerce")
    if "Wellbore name" not in df.columns and "well_short" in df.columns:
        df["Wellbore name"] = df["well_short"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Wells", df["Wellbore name"].nunique() if "Wellbore name" in df.columns else "—")
    c3.metric("Mean ΔP", f"{df['delta_P_measured'].mean():.1f} bar")
    c4.metric("Mean baseline ΔP", f"{df['delta_P_baseline'].mean():.1f} bar")

    atabs = st.tabs(["Per-well summary", "ΔP vs rate", "Distributions"])

    with atabs[0]:
        if "Wellbore name" in df.columns:
            rows = []
            for well, w in df.groupby("Wellbore name"):
                rows.append(
                    {
                        "Well": str(well).replace("15/9-", ""),
                        "Records": len(w),
                        "ΔP mean": round(w["delta_P_measured"].mean(), 1),
                        "Baseline mean": round(w["delta_P_baseline"].mean(), 1),
                        "WC mean": round(w["WC"].mean(), 3) if "WC" in w else None,
                        "q_liq mean": round(w["q_liq"].mean(), 1) if "q_liq" in w else None,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with atabs[1]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(
            df["q_liq"], df["delta_P_measured"], c=df["WC"], cmap="RdYlBu_r", s=12, alpha=0.55
        )
        plt.colorbar(sc, ax=ax, label="Water cut")
        ax.set_xlabel("Liquid rate (Sm³/d)")
        ax.set_ylabel("Measured ΔP (bar)")
        ax.set_title("VLP relationship coloured by water cut")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with atabs[2]:
        feat = st.selectbox("Feature", ["delta_P_measured", "delta_P_baseline", "WC", "GLR", "q_liq"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[feat].dropna(), bins=40, color="#1565C0", alpha=0.8, edgecolor="white")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =====================================================================
# ABOUT (your name, title, institution — hybrid wording)
# =====================================================================
elif page == "ℹ️ About":
    st.markdown(
        """
<div style="text-align: center; padding: 30px 0;">
  <h1 style="color: #1565C0; font-size: 2.5rem;">🛢️ Hybrid VLP Predictor</h1>
  <p style="font-size: 1.2rem; color: #555;">
    A Physics-Informed Hybrid Random Forest Model for<br>
    Vertical Lift Performance Prediction — Volve Field Case Study
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Williams Iwum")
        st.markdown(
            """
**MEng Student**  
Department of Petroleum Engineering  
Federal University of Petroleum Resources  
Effurun, Delta State, Nigeria
"""
        )
    with col2:
        st.markdown("### About this project")
        st.markdown(
            """
This application supports an **MEng dissertation** that develops and evaluates a
**physics-informed hybrid Random Forest** model for tubing pressure-drop prediction
on Equinor’s public **Volve** production dataset.

#### Project title
> *A Physics-Informed Hybrid Random Forest Model for Vertical Lift Performance
> Prediction: A Volve Field Case Study on Cross-Well and Chronological Generalisation*

#### Core idea
Rather than asking Random Forest to learn ΔP from scratch, a simplified
**homogeneous no-slip baseline** is computed first; RF learns only the **residual**.
Final prediction:

\\[\\Delta P_{\\mathrm{hybrid}} = \\Delta P_{\\mathrm{baseline}} + \\mathrm{RF}(r)\\]

Models are compared under **pooled**, **Leave-One-Well-Out**, and **chronological** validation.
"""
        )

    st.markdown("---")
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
<div style="background:#F5F5F5;padding:20px;border-radius:12px;border-left:4px solid #1565C0;">
<h4 style="color:#1565C0;">📊 Data</h4>
<p>Volve open dataset (Equinor) — five producer wells with downhole and wellhead pressures.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
<div style="background:#F5F5F5;padding:20px;border-radius:12px;border-left:4px solid #388E3C;">
<h4 style="color:#388E3C;">🤖 Model</h4>
<p>Homogeneous baseline + Random Forest residual correction (400 trees, max_features=0.5).</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
<div style="background:#F5F5F5;padding:20px;border-radius:12px;border-left:4px solid #E64A19;">
<h4 style="color:#E64A19;">✅ Validation</h4>
<p>Pooled, LOWO, and chronological splits — hybrid vs plain RF vs baseline alone.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Dataset acknowledgement")
    st.info(
        "Volve field dataset — Equinor (2018) open data release. "
        "https://www.equinor.com/energy/volve-data-sharing"
    )
    st.markdown(
        """
<div style="text-align:center;color:#888;font-size:0.9rem;padding:20px;">
<p>© 2026 Williams Iwum | Federal University of Petroleum Resources, Effurun</p>
<p style="font-size:0.8rem;">MEng — Department of Petroleum Engineering</p>
</div>
""",
        unsafe_allow_html=True,
    )

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
Hybrid VLP Predictor | Williams Iwum | Federal University of Petroleum Resources
<br>Physics-Informed Hybrid Random Forest for Vertical Lift Performance Prediction
</div>
""",
    unsafe_allow_html=True,
)
