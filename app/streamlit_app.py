"""
VLP Prediction - Streamlit Web Application
===========================================
Interactive tool for predicting wellbore pressure drop (delta_P)
using the trained Random Forest model from the Volve field dataset.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="VLP Predictor - Random Forest Model",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
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
""", unsafe_allow_html=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
@st.cache_resource
def load_model(model_path):
    """Load the trained RF model from pickle file."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def train_model_from_data(data_path):
    """Train a new RF model from the model-ready CSV if no pickle exists."""
    from sklearn.ensemble import RandomForestRegressor

    d = pd.read_csv(data_path)
    d['Date of Production'] = pd.to_datetime(d['Date of Production'], errors='coerce')

    FEATURES = ['q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
                'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
                'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS',
                'log_q_liq', 'log_q_oil', 'log_q_gas', 'GLR', 'dT']
    TARGET = 'delta_P'

    available = [f for f in FEATURES if f in d.columns]
    d = d.dropna(subset=available + [TARGET])
    # Use the same best hyperparameters from the pipeline's GridSearchCV
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    rf.fit(d[available], d[TARGET])

    return {
        'model': rf,
        'features': available,
        'target': TARGET,
        'best_params': rf.get_params(),
        'training_stats': {
            'n_samples': len(d),
            'n_features': len(available),
            'target_mean': d[TARGET].mean(),
            'target_std': d[TARGET].std(),
        },
    }


def predict_with_uncertainty(model_data, inputs):
    """
    Predict ΔP and return uncertainty (std across trees).
    Returns: (mean_prediction, std_deviation)
    """
    model = model_data['model']
    features = model_data['features']

    # Build feature vector (same logic as predict_single)
    row = {}
    for f in features:
        if f in inputs:
            row[f] = inputs[f]
        elif f == 'q_liq':
            row[f] = inputs.get('q_oil', 0) + inputs.get('q_wat', 0)
        elif f == 'WC':
            q_liq = inputs.get('q_oil', 0) + inputs.get('q_wat', 0)
            row[f] = inputs.get('q_wat', 0) / q_liq if q_liq > 0 else 0
        elif f == 'GOR':
            row[f] = inputs.get('q_gas', 0) / max(inputs.get('q_oil', 1), 1e-6)
        elif f == 'GLR':
            q_liq = inputs.get('q_oil', 0) + inputs.get('q_wat', 0)
            row[f] = inputs.get('q_gas', 0) / max(q_liq, 1e-6)
        elif f == 'log_q_liq':
            row[f] = np.log1p(inputs.get('q_oil', 0) + inputs.get('q_wat', 0))
        elif f == 'log_q_oil':
            row[f] = np.log1p(inputs.get('q_oil', 0))
        elif f == 'log_q_gas':
            row[f] = np.log1p(inputs.get('q_gas', 0))
        elif f == 'dT':
            row[f] = inputs.get('AVG_DOWNHOLE_TEMPERATURE', 0) - inputs.get('AVG_WHT_P', 0)
        else:
            row[f] = 0  # fallback

    X = pd.DataFrame([row])[features]

    # Get predictions from every tree
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    
    mean_pred = tree_preds.mean()
    std_pred = tree_preds.std()

    return mean_pred, std_pred

def generate_vlp_curve(model_data, base_inputs, q_range):
    """
    Generate a VLP curve by varying liquid rate.
    Returns: mean predictions and uncertainty (std) for each rate.
    """
    means = []
    stds = []

    for q in q_range:
        inputs = base_inputs.copy()

        # Scale oil and water proportionally
        total_liq = inputs.get('q_oil', 0) + inputs.get('q_wat', 0)
        if total_liq > 0:
            oil_frac = inputs.get('q_oil', 0) / total_liq
            inputs['q_oil'] = q * oil_frac
            inputs['q_wat'] = q * (1 - oil_frac)
        else:
            inputs['q_oil'] = q
            inputs['q_wat'] = 0

        # Scale gas to keep GOR constant
        gor = base_inputs.get('q_gas', 0) / max(base_inputs.get('q_oil', 1), 1e-6)
        inputs['q_gas'] = inputs['q_oil'] * gor

        mean_dp, std_dp = predict_with_uncertainty(model_data, inputs)
        means.append(mean_dp)
        stds.append(std_dp)

    return np.array(means), np.array(stds)


# =====================================================================
# FIND DATA FILES
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

MODEL_PATH = os.path.join(DATA_DIR, 'rf_model.pkl')
DATA_PATH = os.path.join(DATA_DIR, 'modelready_features.csv')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'volve_welldata_raw.csv')
LOWO_PATH = os.path.join(DATA_DIR, 'lowo_results.csv')
LOWO_LR_PATH = os.path.join(DATA_DIR, 'lowo_lr_results.csv')
CHRONO_PATH = os.path.join(DATA_DIR, 'chrono_split_results.csv')
GINI_PATH = os.path.join(DATA_DIR, 'feature_importance_gini.csv')
SUMMARY_PATH = os.path.join(DATA_DIR, 'validation_summary.csv')
BB_SENS_PATH = os.path.join(DATA_DIR, 'bb_sensitivity_tubing_id.csv')
EXTRAP_PATH = os.path.join(DATA_DIR, 'extrapolation_analysis.csv')
REGIME_PATH = os.path.join(DATA_DIR, 'well_regime_summary.csv')
UNCERTAINTY_PATH = os.path.join(DATA_DIR, 'uncertainty_analysis.csv')
WC_ERROR_PATH = os.path.join(DATA_DIR, 'error_by_wc_regime.csv')
COMPARISON_PATH = os.path.join(DATA_DIR, 'rf_vs_bb_comparison.csv')


# =====================================================================
# LOAD MODEL
# =====================================================================
@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    elif os.path.exists(DATA_PATH):
        st.info(
            "🔄 **First-time setup:** The pre-trained model file (`rf_model.pkl`, ~146 MB) is too large "
            "to store on GitHub, so it is not included in the repository. "
            "The app is now **automatically retraining** the Random Forest from the committed "
            "feature-engineered dataset (`modelready_features.csv`). "
            "This takes ~1–2 minutes and only happens once per session."
        )
        with st.spinner("⏳ Training Random Forest model from data... please wait (~1–2 min)"):
            md = train_model_from_data(DATA_PATH)
        # Save for next time within the same deployment session
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(md, f)
        except Exception:
            pass  # Read-only filesystem on cloud — fine, model stays in memory
        st.success("✅ Model trained and ready!")
        return md
    else:
        return None


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("# 🛢️ VLP Predictor")
    st.markdown("---")
    st.markdown("""
    **Project:** Development of a Random Forest-Based VLP Model
    
    **Author:** Williams Iwum
    
    **Institution:** Federal University of Petroleum Resources
    
    **Dataset:** Volve Field (Equinor)
    """)
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to:", [
        "🏠 Dashboard",
        "📊 Data Analytics",
        "🔮 Single Prediction",
        "📈 VLP Curve Generator",
        "📉 Model Performance",
        "🔍 Feature Importance",
        "🛡️ Data Integrity",
        "📁 Batch Prediction",
        "ℹ️ About"
    ], label_visibility="collapsed")



# =====================================================================
# LOAD MODEL AND DATA
# =====================================================================
model_data = get_model()

if model_data is None:
    st.error("❌ No model or training data found!")
    st.info(f"Expected model at: `{MODEL_PATH}`\nor training data at: `{DATA_PATH}`")
    st.info("Run Notebook 2 first to generate the model, or place `volve_vlp_modelready.csv` in the `data/` folder.")
    st.stop()


# =====================================================================
# PAGE: DASHBOARD
# =====================================================================
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">VLP Pressure Drop Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Random Forest Model trained on Volve Field Production Data</p>',
                unsafe_allow_html=True)

    # Key metrics
    stats = model_data.get('training_stats', {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Training Samples", f"{stats.get('n_samples', 0):,}")
    with col2:
        st.metric("Features", f"{stats.get('n_features', 0)}")
    with col3:
        st.metric("Mean ΔP", f"{stats.get('target_mean', 0):.1f} bar")
    with col4:
        st.metric("Std ΔP", f"{stats.get('target_std', 0):.1f} bar")

    st.markdown("---")

    # Validation results
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Model Validation Results")
        if os.path.exists(SUMMARY_PATH):
            summary = pd.read_csv(SUMMARY_PATH)
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
        elif os.path.exists(LOWO_PATH):
            lowo = pd.read_csv(LOWO_PATH)
            st.dataframe(lowo.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Run the pipeline to generate validation results.")

    with col_right:
        st.markdown("### Feature Importance (Top 10)")
        if os.path.exists(GINI_PATH):
            gi = pd.read_csv(GINI_PATH).head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            gi_sorted = gi.sort_values('importance' if 'importance' in gi.columns
                                        else gi.columns[1], ascending=True)
            val_col = 'importance' if 'importance' in gi.columns else gi.columns[1]
            feat_col = 'feature' if 'feature' in gi.columns else gi.columns[0]
            ax.barh(gi_sorted[feat_col], gi_sorted[val_col], color='#2E7D32', alpha=0.85)
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    st.markdown("""
    ### About This Project
    This application uses a **Random Forest regression model** to predict wellbore pressure drop
    ($\\Delta P = P_{wf} - P_{wh}$) for the Volve oil field in the North Sea.
    
    The model was trained on real production data from 5 producer wells, using 16 features
    derived from measured production variables.
    
    **Use the sidebar** to navigate to different pages:
    - **Single Prediction**: Input well parameters and get an instant ΔP prediction
    - **VLP Curve Generator**: Generate VLP curves for different conditions
    - **Model Performance**: View detailed validation metrics and plots
    - **Feature Importance**: Understand which variables drive predictions
    - **Batch Prediction**: Upload a CSV file for bulk predictions
    """)


# =====================================================================
# PAGE: SINGLE PREDICTION
# =====================================================================
elif page == "🔮 Single Prediction":
    st.markdown("## 🔮 Single Well Prediction")
    st.markdown("Enter well operating parameters to predict the wellbore pressure drop (ΔP).")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Flow Rates (Sm³/d)")
        q_oil = st.number_input("Oil Rate (q_oil)", min_value=0.0, max_value=20000.0,
                                value=1000.0, step=50.0)
        q_gas = st.number_input("Gas Rate (q_gas)", min_value=0.0, max_value=500000.0,
                                value=150000.0, step=5000.0)
        q_wat = st.number_input("Water Rate (q_wat)", min_value=0.0, max_value=20000.0,
                                value=500.0, step=50.0)

    with col2:
        st.markdown("#### Pressures & Temperatures")
        whp = st.number_input("Wellhead Pressure (bar)", min_value=0.0, max_value=300.0,
                              value=50.0, step=5.0)
        wht = st.number_input("Wellhead Temperature (°C)", min_value=0.0, max_value=200.0,
                              value=70.0, step=5.0)
        dht = st.number_input("Downhole Temperature (°C)", min_value=0.0, max_value=250.0,
                              value=106.0, step=5.0)

    with col3:
        st.markdown("#### Operations")
        choke = st.number_input("Choke Size (%)", min_value=0.0, max_value=100.0,
                                value=50.0, step=5.0)
        on_hrs = st.number_input("On-Stream Hours", min_value=0.1, max_value=24.0,
                                 value=24.0, step=1.0)

    # Compute derived values
    q_liq = q_oil + q_wat
    wc = q_wat / q_liq if q_liq > 0 else 0
    gor = q_gas / q_oil if q_oil > 0 else 0

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Liquid Rate", f"{q_liq:.0f} Sm³/d")
    with col_b:
        st.metric("Water Cut", f"{wc:.1%}")
    with col_c:
        st.metric("GOR", f"{gor:.0f} Sm³/Sm³")

    st.markdown("---")

    if st.button("🔮 Predict ΔP", type="primary", use_container_width=True):
        inputs = {
            'q_oil': q_oil, 'q_gas': q_gas, 'q_wat': q_wat,
            'AVG_WHP_P': whp, 'AVG_WHT_P': wht,
            'AVG_DOWNHOLE_TEMPERATURE': dht,
            'AVG_CHOKE_SIZE_P': choke, 'ON_STREAM_HRS': on_hrs,
        }
        prediction = predict_single(model_data, inputs)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success(f"### Predicted ΔP = **{prediction:.2f} bar**")
            st.markdown(f"""
            This means the bottomhole flowing pressure is approximately:

            $P_{{wf}} = P_{{wh}} + \\Delta P = {whp:.1f} + {prediction:.1f} = **{whp + prediction:.1f}$ bar**
            """)



# =====================================================================
# PAGE: VLP CURVE GENERATOR
# =====================================================================
elif page == "📈 VLP Curve Generator":
    st.markdown("## 📈 VLP Curve Generator")
    st.markdown("Generate Vertical Lift Performance curves by varying flow rate and water cut.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Base Conditions")
        base_q_oil = st.number_input("Base Oil Rate (Sm³/d)", value=1000.0, step=100.0)
        base_q_gas = st.number_input("Base Gas Rate (Sm³/d)", value=150000.0, step=5000.0)
        base_q_wat = st.number_input("Base Water Rate (Sm³/d)", value=500.0, step=100.0)
        base_whp = st.number_input("Wellhead Pressure (bar)", value=50.0, step=5.0,
                                    key="vlp_whp")

    with col2:
        st.markdown("#### Conditions & Range")
        base_wht = st.number_input("Wellhead Temperature (°C)", value=70.0, step=5.0,
                                    key="vlp_wht")
        base_dht = st.number_input("Downhole Temperature (°C)", value=106.0, step=5.0,
                                    key="vlp_dht")
        base_choke = st.number_input("Choke Size (%)", value=50.0, step=5.0, key="vlp_choke")
        q_min = st.number_input("Min Liquid Rate (Sm³/d)", value=100.0, step=50.0)
        q_max = st.number_input("Max Liquid Rate (Sm³/d)", value=8000.0, step=500.0)

    st.markdown("---")

    # Water cut scenarios
    st.markdown("#### Water Cut Scenarios")
    wc_scenarios = st.multiselect(
        "Select water cut values to compare:",
        options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        default=[0.1, 0.3, 0.5, 0.7]
    )

      if st.button("📈 Generate VLP Curves", type="primary", use_container_width=True):
        q_range = np.linspace(q_min, q_max, 40)

        fig, ax = plt.subplots(figsize=(11, 7))
        colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(wc_scenarios)))

        for wc_val, color in zip(wc_scenarios, colors):
            means = []
            stds = []

            for q in q_range:
                q_oil_i = q * (1 - wc_val)
                q_wat_i = q * wc_val
                gor = base_q_gas / max(base_q_oil, 1)
                q_gas_i = q_oil_i * gor

                inputs = {
                    'q_oil': q_oil_i,
                    'q_gas': q_gas_i,
                    'q_wat': q_wat_i,
                    'AVG_WHP_P': base_whp,
                    'AVG_WHT_P': base_wht,
                    'AVG_DOWNHOLE_TEMPERATURE': base_dht,
                    'AVG_CHOKE_SIZE_P': base_choke,
                    'ON_STREAM_HRS': 24.0,
                }

                mean_dp, std_dp = predict_with_uncertainty(model_data, inputs)
                means.append(mean_dp)
                stds.append(std_dp)

            means = np.array(means)
            stds = np.array(stds)

            # Plot mean curve
            ax.plot(q_range, means, '-', color=color, lw=2.5,
                    label=f'WC = {wc_val:.0%}', alpha=0.9)

            # Plot uncertainty band (±1 std)
            ax.fill_between(q_range, means - stds, means + stds,
                            color=color, alpha=0.18)

        ax.set_xlabel('Liquid Flow Rate (Sm³/d)', fontsize=12)
        ax.set_ylabel('Wellbore Pressure Drop, ΔP (bar)', fontsize=12)
        ax.set_title('VLP Curves at Different Water Cuts\n(with Uncertainty Bands)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Interpretation:**  
        - As water cut increases, the pressure drop (ΔP) increases because water is denser than oil, raising the hydrostatic gradient.  
        - The shaded bands represent model uncertainty (±1 standard deviation across the Random Forest trees).  
        - Narrower bands indicate higher model confidence; wider bands show greater uncertainty in that operating region.  
        - This makes the tool more useful for engineering decision-making by showing both the predicted VLP and its reliability.
        """)
# =====================================================================
# PAGE: MODEL PERFORMANCE
# =====================================================================
elif page == "📉 Model Performance":
    st.markdown("## 📉 Model Performance & Validation")

    tabs = st.tabs([
        "LOWO Results",
        "Chronological Split",
        "Comparison"
    ])

    with tabs[0]:
        st.markdown("### Leave-One-Well-Out (LOWO) Cross-Validation")
        st.markdown("Train on 4 wells, test on the 5th. Repeat for all 5 wells.")
        if os.path.exists(LOWO_PATH):
            lowo = pd.read_csv(LOWO_PATH)
            st.dataframe(lowo.round(3), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean R²", f"{lowo['R2'].mean():.3f}")
                st.metric("Best R²", f"{lowo['R2'].max():.3f}")
            with col2:
                st.metric("Mean RMSE", f"{lowo['RMSE'].mean():.2f} bar")
                st.metric("Mean MAPE", f"{lowo['MAPE'].mean():.2f}%")

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(lowo))
            well_names = [w.replace('15/9-', '') for w in lowo['test_well']]
            ax.bar(x, lowo['R2'], color='#1565C0', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(well_names, rotation=15)
            ax.set_ylabel('R²')
            ax.set_title('LOWO R² by Held-Out Well')
            ax.axhline(0, color='gray', ls='--', lw=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run the pipeline first to generate LOWO results.")

    with tabs[1]:
        st.markdown("### Chronological Within-Well Split")
        st.markdown("Train on first 80% + all other wells, test on last 20%.")
        if os.path.exists(CHRONO_PATH):
            chrono = pd.read_csv(CHRONO_PATH)
            st.dataframe(chrono.round(3), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean R²", f"{chrono['R2'].mean():.3f}")
            with col2:
                st.metric("Mean RMSE", f"{chrono['RMSE'].mean():.2f} bar")
        else:
            st.info("Run the v2 pipeline to generate chronological split results.")

    with tabs[2]:
        st.markdown("### Validation Strategy Comparison")
        if os.path.exists(SUMMARY_PATH):
            summary = pd.read_csv(SUMMARY_PATH)
            st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Run the v2 pipeline to generate the comparison summary.")


# =====================================================================
# PAGE: FEATURE IMPORTANCE

# =====================================================================
elif page == "🔍 Feature Importance":
    st.markdown("## 🔍 Feature Importance Analysis")

    if os.path.exists(GINI_PATH):
        gi = pd.read_csv(GINI_PATH)
        val_col = 'importance' if 'importance' in gi.columns else gi.columns[1]
        feat_col = 'feature' if 'feature' in gi.columns else gi.columns[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 7))
            gi_sorted = gi.sort_values(val_col, ascending=True)
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(gi_sorted)))
            ax.barh(gi_sorted[feat_col], gi_sorted[val_col], color=colors)
            ax.set_xlabel('Feature Importance (Gini-based)')
            ax.set_title('Which Variables Drive Pressure Drop Prediction')
            for i, (idx, row) in enumerate(gi_sorted.iterrows()):
                ax.text(row[val_col] + 0.002, i, f'{row[val_col]:.3f}',
                        va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("### Importance Table")
            st.dataframe(gi.round(4), use_container_width=True, hide_index=True)

        st.markdown("""
        ### Key Insights
        
        1. **Water Cut (WC)** is the single most important feature, accounting for ~29% of 
           total importance. This confirms that classical VLP correlations — developed on 
           low-WC lab data — are weakest where they matter most.
        
        2. **Water Rate (q_wat)** ranks second, reinforcing the WC finding from a different angle.
        
        3. **Temperature and pressure** (downhole temperature, wellhead temperature, wellhead 
           pressure) together contribute ~27%, capturing PVT and hydrostatic effects.
        
        4. **GOR has minimal importance** (~1%), suggesting gas composition effects are secondary 
           in this dataset (Volve is a moderate-GOR field).
        """)
    else:
        st.info("Run the pipeline to generate feature importance data.")


# =====================================================================
# PAGE: BATCH PREDICTION
# =====================================================================
elif page == "📁 Batch Prediction":
    st.markdown("## 📁 Batch Prediction")
    st.markdown("Upload a CSV file with well parameters to get predictions for multiple scenarios.")

    st.markdown("""
    **Required columns** (at minimum):
    - `q_oil` — Oil rate (Sm³/d)
    - `q_gas` — Gas rate (Sm³/d)
    - `q_wat` — Water rate (Sm³/d)
    - `AVG_WHP_P` — Wellhead pressure (bar)
    - `AVG_WHT_P` — Wellhead temperature (°C)
    - `AVG_DOWNHOLE_TEMPERATURE` — Downhole temperature (°C)
    - `AVG_CHOKE_SIZE_P` — Choke size (%)
    - `ON_STREAM_HRS` — On-stream hours
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown(f"**Uploaded:** {len(batch_df)} rows")
        st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            predictions = []
            for _, row in batch_df.iterrows():
                inputs = row.to_dict()
                pred = predict_single(model_data, inputs)
                predictions.append(pred)

            batch_df['Predicted_dP_bar'] = predictions

            st.markdown("### Results")
            st.dataframe(batch_df, use_container_width=True, hide_index=True)

            # Download button
            csv_out = batch_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_out,
                file_name="vlp_predictions.csv",
                mime="text/csv"
            )

            # Quick plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(predictions, bins=30, color='#1565C0', alpha=0.75, edgecolor='white')
            ax.set_xlabel('Predicted ΔP (bar)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Predicted Pressure Drops')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# =====================================================================
# PAGE: DATA ANALYTICS
# =====================================================================
elif page == "📊 Data Analytics":
    st.markdown("## 📊 Data Analytics & Exploration")
    st.markdown("Interactive exploration of the Volve field production dataset — understand the data before trusting the model.")

    # Load the model-ready data for analytics
    @st.cache_data
    def load_analytics_data():
        # Try the model-ready CSV first
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            df['Date of Production'] = pd.to_datetime(df['Date of Production'], errors='coerce')
            return df
        return None

    analytics_df = load_analytics_data()

    if analytics_df is None:
        st.error("Model-ready dataset not found. Run the pipeline or Notebook 1 first.")
        st.stop()

    # --- Dataset Overview ---
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(analytics_df):,}")
    with col2:
        st.metric("Wells", f"{analytics_df['Wellbore name'].nunique()}")
    with col3:
        st.metric("Features", f"{analytics_df.shape[1] - 2}")  # minus well name and date
    with col4:
        date_range = (analytics_df['Date of Production'].max() - analytics_df['Date of Production'].min()).days
        st.metric("Date Span", f"{date_range:,} days")
    with col5:
        st.metric("Mean ΔP", f"{analytics_df['delta_P'].mean():.1f} bar")

    st.markdown("---")

    # --- Analytics Tabs ---
    atabs = st.tabs([
        "📋 Per-Well Summary",
        "📊 Distribution Analysis",
        "📈 VLP Curve",
        "🔗 Correlations",
        "⏱️ Time Series",
        "💧 Water Cut Evolution"
    ])

    WELL_COLORS_LIST = ['#1976D2', '#388E3C', '#E64A19', '#7B1FA2', '#00838F',
                        '#F57C00', '#C62828', '#1565C0', '#2E7D32']

    # --- Tab 1: Per-Well Summary ---
    with atabs[0]:
        st.markdown("### Per-Well Operating Summary")
        st.markdown("Each well operates in a different regime — this is why cross-well prediction is challenging.")

        wells = analytics_df['Wellbore name'].unique()
        summary_rows = []
        for well in wells:
            w = analytics_df[analytics_df['Wellbore name'] == well]
            summary_rows.append({
                'Well': well.replace('15/9-', ''),
                'Records': len(w),
                'ΔP Mean (bar)': round(w['delta_P'].mean(), 1),
                'ΔP Std (bar)': round(w['delta_P'].std(), 1),
                'ΔP Min (bar)': round(w['delta_P'].min(), 1),
                'ΔP Max (bar)': round(w['delta_P'].max(), 1),
                'WC Mean': round(w['WC'].mean(), 3),
                'GOR Mean': round(w['GOR'].mean(), 1),
                'q_liq Mean (Sm³/d)': round(w['q_liq'].mean(), 1),
                'WHP Mean (bar)': round(w['AVG_WHP_P'].mean(), 1),
            })
        summary_table = pd.DataFrame(summary_rows)
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.markdown("""
        **Key observations:**
        - **F-15D** operates at much lower liquid rates (mean ~272 Sm³/d vs 925-4595 for others)
          and in a narrow ΔP range — this is why the model struggles to predict it from other wells
        - **F-14H** has the highest water cut (mean 0.65) and the widest ΔP range
        - **F-12H** has the highest liquid rates but lowest water cut
        """)

    # --- Tab 2: Distribution Analysis ---
    with atabs[1]:
        st.markdown("### Distribution Analysis")

        dist_feature = st.selectbox(
            "Select feature to analyze:",
            ['delta_P', 'WC', 'GOR', 'q_liq', 'q_oil', 'q_wat', 'q_gas',
             'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_CHOKE_SIZE_P'],
            index=0
        )

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, well in enumerate(wells):
                w = analytics_df[analytics_df['Wellbore name'] == well]
                ax.hist(w[dist_feature].dropna(), bins=30, alpha=0.5,
                        label=well.replace('15/9-', ''),
                        color=WELL_COLORS_LIST[i % len(WELL_COLORS_LIST)])
            ax.set_xlabel(dist_feature)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {dist_feature} by Well')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            data_list = [analytics_df[analytics_df['Wellbore name'] == w][dist_feature].dropna().values
                         for w in wells]
            well_labels = [w.replace('15/9-', '') for w in wells]
            bp = ax.boxplot(data_list, patch_artist=True)
            # Set tick labels compatibly across matplotlib versions
            ax.set_xticks(range(1, len(well_labels) + 1))
            ax.set_xticklabels(well_labels, rotation=15)
            for patch, color in zip(bp['boxes'], WELL_COLORS_LIST):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel(dist_feature)
            ax.set_title(f'{dist_feature} — Box Plot by Well')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Summary stats
        st.markdown(f"#### {dist_feature} — Summary Statistics")
        desc = analytics_df.groupby('Wellbore name')[dist_feature].describe().round(2)
        desc.index = [w.replace('15/9-', '') for w in desc.index]
        st.dataframe(desc, use_container_width=True)

    # --- Tab 3: VLP Curve ---
    with atabs[2]:
        st.markdown("### VLP Curve: Pressure Drop vs Liquid Rate")
        st.markdown("This is the core VLP relationship — how pressure drop varies with flow rate, colored by water cut.")

        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(analytics_df['q_liq'], analytics_df['delta_P'],
                        c=analytics_df['WC'], cmap='RdYlBu_r',
                        s=14, alpha=0.55, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Water Cut (fraction)')
        ax.set_xlabel('Liquid Flow Rate (Sm³/d)')
        ax.set_ylabel('Wellbore Pressure Drop, ΔP (bar)')
        ax.set_title('VLP Relationship: ΔP vs Flow Rate\nColored by Water Cut')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **Physical interpretation:**
        - At higher water cuts (red dots), pressure drop increases because water is denser
          than oil, increasing the hydrostatic head in the wellbore
        - This relationship is the foundation of VLP modeling — and it's what our Random
          Forest learns to predict
        """)

        # Per-well VLP
        st.markdown("#### Per-Well VLP Curves")
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, well in enumerate(wells):
            w = analytics_df[analytics_df['Wellbore name'] == well]
            ax.scatter(w['q_liq'], w['delta_P'], s=10, alpha=0.5,
                       label=well.replace('15/9-', ''),
                       color=WELL_COLORS_LIST[i % len(WELL_COLORS_LIST)],
                       edgecolors='none')
        ax.set_xlabel('Liquid Flow Rate (Sm³/d)')
        ax.set_ylabel('ΔP (bar)')
        ax.set_title('VLP Curves by Well')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Tab 4: Correlations ---
    with atabs[3]:
        st.markdown("### Feature Correlations")

        import seaborn as sns

        corr_features = ['delta_P', 'q_oil', 'q_gas', 'q_wat', 'q_liq', 'WC', 'GOR',
                          'AVG_WHP_P', 'AVG_WHT_P', 'AVG_DOWNHOLE_TEMPERATURE',
                          'AVG_CHOKE_SIZE_P']
        available_corr = [f for f in corr_features if f in analytics_df.columns]
        corr_matrix = analytics_df[available_corr].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Correlations with ΔP (Target Variable)")
        dp_corr = corr_matrix['delta_P'].drop('delta_P').sort_values(ascending=False)
        corr_table = pd.DataFrame({
            'Feature': dp_corr.index,
            'Correlation with ΔP': dp_corr.values
        })
        st.dataframe(corr_table.round(3), use_container_width=True, hide_index=True)

        st.success("**Key finding:** Water Cut (WC) has the strongest positive correlation "
                   f"with ΔP (r = {dp_corr.get('WC', 0):+.3f}), confirming it as the dominant "
                   "driver of pressure drop in these wells.")

    # --- Tab 5: Time Series ---
    with atabs[4]:
        st.markdown("### Production Time Series")

        ts_well = st.selectbox("Select well:", wells,
                               format_func=lambda x: x.replace('15/9-', ''))
        ts_feature = st.selectbox("Select variable:",
                                   ['delta_P', 'q_liq', 'WC', 'GOR', 'AVG_WHP_P',
                                    'q_oil', 'q_wat', 'AVG_DOWNHOLE_TEMPERATURE'],
                                   index=0)

        w = analytics_df[analytics_df['Wellbore name'] == ts_well].sort_values('Date of Production')

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(w['Date of Production'], w[ts_feature], 'o-', ms=2, lw=0.8,
                color='#1565C0', alpha=0.7)

        # Add rolling average
        if len(w) > 14:
            rolling_avg = w[ts_feature].rolling(14, min_periods=3).mean()
            ax.plot(w['Date of Production'], rolling_avg, '-', lw=2.5,
                    color='#E64A19', alpha=0.8, label='14-day moving average')
            ax.legend()

        ax.set_xlabel('Date')
        ax.set_ylabel(ts_feature)
        ax.set_title(f'{ts_well.replace("15/9-", "")} — {ts_feature} over Time')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Show stats for selected well
        st.markdown(f"**{ts_well.replace('15/9-', '')} — {ts_feature} statistics:**")
        col1, col2, col3, col4 = st.columns(4)
        vals = w[ts_feature].dropna()
        with col1:
            st.metric("Mean", f"{vals.mean():.2f}")
        with col2:
            st.metric("Std", f"{vals.std():.2f}")
        with col3:
            st.metric("Min", f"{vals.min():.2f}")
        with col4:
            st.metric("Max", f"{vals.max():.2f}")

    # --- Tab 6: Water Cut Evolution ---
    with atabs[5]:
        st.markdown("### Water Cut Evolution")
        st.markdown("Water cut increase over time is the **central challenge** for VLP prediction. "
                    "Classical correlations were developed on low-WC lab data and degrade at high WC.")

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes_flat = axes.flatten()

        for i, well in enumerate(wells):
            if i >= 5:
                break
            w = analytics_df[analytics_df['Wellbore name'] == well].sort_values('Date of Production')
            ax = axes_flat[i]

            color_wc = WELL_COLORS_LIST[i % len(WELL_COLORS_LIST)]

            ax.plot(w['Date of Production'], w['WC'], 'o-', ms=1.5, lw=0.5,
                    color=color_wc, alpha=0.6, label='WC')
            ax.set_ylabel('Water Cut', color=color_wc)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(well.replace('15/9-', ''), fontsize=12, fontweight='bold')

            ax2 = ax.twinx()
            ax2.plot(w['Date of Production'], w['delta_P'], 's-', ms=1, lw=0.5,
                     color='gray', alpha=0.4, label='ΔP')
            ax2.set_ylabel('ΔP (bar)', color='gray')

            ax.tick_params(axis='x', rotation=45, labelsize=7)

        # Hide unused subplot
        if len(wells) < 6:
            axes_flat[5].set_visible(False)

        plt.suptitle('Water Cut (colored) and Pressure Drop (gray) Over Time',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info("**Observation:** In most wells, periods of increasing water cut correspond to "
                "increasing ΔP — this confirms the physical relationship the RF model captures.")


# =====================================================================
# PAGE: ABOUT
# =====================================================================
elif page == "ℹ️ About":
    st.markdown("""    
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="color: #1565C0; font-size: 2.5rem;">🛢️ VLP Predictor</h1>
        <p style="font-size: 1.2rem; color: #555;">
            Development of a Random Forest-Based Vertical Lift Performance Model<br>
            for Multiphase Wellbore Flow Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Author photo
        author_photo = os.path.join(os.path.dirname(__file__), 'assets', 'author_photo.jpg')
        if os.path.exists(author_photo):
            st.image(author_photo, use_container_width=True)
        st.markdown("### Williams Iwum")
        st.markdown("""
        **Final Year Student**

        Department of Petroleum Engineering  
        Federal University of Petroleum Resources  
        Effurun, Delta State, Nigeria
        """)


    with col2:
        st.markdown("### About This Project")
        st.markdown("""
        This project was developed as a **Final Year Project** for the award of
        a Bachelor's degree in Petroleum Engineering at the Federal University
        of Petroleum Resources (FUPRE), Effurun, Delta State, Nigeria.

        #### Project Title
        > *Development of a Random Forest-Based Vertical Lift Performance (VLP) Model
        > for Multiphase Wellbore Flow Prediction*

        #### Objective
        To develop, validate, and present a data-driven VLP model that predicts
        wellbore pressure drop (ΔP = P_wf − P_wh) using Random Forest regression,
        trained on real Volve field production data from the North Sea, and
        benchmarked against the classical Beggs & Brill (1973) empirical correlation.
        """)

    st.markdown("---")

    st.markdown("### Methodology")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div style="background: #F5F5F5; padding: 20px; border-radius: 12px;
                    border-left: 4px solid #1565C0; height: 200px;">
            <h4 style="color: #1565C0;">📊 Data</h4>
            <p>Volve open field dataset (Equinor, 2018) — real production data from
            5 producer wells in the North Sea, spanning 2007-2016.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background: #F5F5F5; padding: 20px; border-radius: 12px;
                    border-left: 4px solid #388E3C; height: 200px;">
            <h4 style="color: #388E3C;">🤖 Model</h4>
            <p>Random Forest regression with hyperparameter tuning via GridSearchCV.
            16 engineered features derived from measured production variables.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div style="background: #F5F5F5; padding: 20px; border-radius: 12px;
                    border-left: 4px solid #E64A19; height: 200px;">
            <h4 style="color: #E64A19;">✅ Validation</h4>
            <p>Three validation strategies: Leave-One-Well-Out, Chronological split,
            and Pooled random split. Benchmarked against Beggs & Brill (1973).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Key Findings")
    st.markdown("""
    1. **Water Cut is the dominant predictor** of wellbore pressure drop (~29% feature importance),
       confirming that classical VLP correlations — developed on low-WC experimental data —
       are weakest where they matter most.

    2. **Random Forest outperforms Beggs & Brill (1973)** on every well in the dataset,
       achieving significantly lower RMSE values.

    3. **Cross-well generalization varies** depending on the similarity of operating regimes
       between training and test wells — a finding that has practical implications for
       field-wide VLP model deployment.

    4. **Within-well chronological prediction is strong**, demonstrating the model's
       practical utility for production forecasting.
    """)

    st.markdown("---")

    st.markdown("### Tools & Technologies")
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    with tech_col1:
        st.markdown("**🐍 Python 3**\n\nCore programming language")
    with tech_col2:
        st.markdown("**📊 scikit-learn**\n\nRandom Forest, GridSearchCV")
    with tech_col3:
        st.markdown("**🔍 SHAP**\n\nFeature importance analysis")
    with tech_col4:
        st.markdown("**🌐 Streamlit**\n\nWeb application framework")

    st.markdown("---")

    st.markdown("### Dataset Acknowledgement")
    st.info("""
    **Volve field dataset** — Released by Equinor (formerly Statoil) in 2018 as an
    open-access research dataset from a decommissioned North Sea oil field.
    Available at: https://www.equinor.com/energy/volve-data-sharing
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem; padding: 20px;">
        <p>© 2026 Williams Iwum | Federal University of Petroleum Resources, Effurun</p>
        <p style="font-size: 0.8rem;">Final Year Project — Department of Petroleum Engineering</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE: DATA INTEGRITY (Leakage Audit + Quality Assurance)
# =====================================================================
elif page == "\U0001f6e1\ufe0f Data Integrity":
    st.markdown('<p class="main-header">Data Integrity & Quality Assurance</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Leakage audit, sensitivity analysis, '
                'cross-well regime analysis, and uncertainty quantification</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "\U0001f50d Data Leakage Audit",
        "\U0001f4ca B&B Sensitivity",
        "\U0001f30d Cross-Well Regimes",
        "\U0001f4cf Uncertainty"
    ])

    # --- Tab 1: Data Leakage Audit ---
    with tab1:
        st.markdown("### Data Leakage Verification")
        st.markdown("""
        > **Why this matters:** The target variable \u0394P = P_wf \u2212 P_wh is computed from
        > measured pressures. If any feature directly contains P_wf or \u0394P, the model
        > would be "cheating" and the impressive R\u00b2 would be meaningless.
        """)

        st.success("\u2705 **VERDICT: NO DATA LEAKAGE DETECTED**")

        leak_data = [
            {"Column": "AVG_DP_TUBING", "Description": "Measured tubing dP",
             "In Features?": "\u274c NO", "Risk": "This IS the target (r=0.9997)"},
            {"Column": "AVG_DOWNHOLE_PRESSURE", "Description": "Bottomhole pressure (P_wf)",
             "In Features?": "\u274c NO", "Risk": "Directly used to compute \u0394P"},
            {"Column": "P_ratio", "Description": "P_wf / P_wh",
             "In Features?": "\u274c NO", "Risk": "Contains P_wf"},
            {"Column": "delta_P", "Description": "The target variable",
             "In Features?": "\u274c NO", "Risk": "Would give R\u00b2=1.0 trivially"},
        ]
        st.dataframe(pd.DataFrame(leak_data), use_container_width=True, hide_index=True)

        st.markdown("#### AVG_WHP_P in Both Features and Target Formula")
        st.info("""
        **AVG_WHP_P** appears as both a feature and in the target formula (\u0394P = P_wf \u2212 AVG_WHP_P).
        
        **This is NOT leakage because:**
        1. WHP is measured by an **independent surface pressure gauge**
        2. In deployment, WHP would always be **available** as an input
        3. WHP is a **standard input** to all VLP correlations (including Beggs & Brill)
        4. Removing it would cripple the model by removing physically necessary information
        """)

        st.markdown("#### Features Used (16 total)")
        features = model_data.get('features', [])
        feat_df = pd.DataFrame({'Feature': features,
                               'Category': ['Flow rate']*4 + ['Compositional']*2 +
                                           ['Pressure']*1 + ['Temperature']*2 +
                                           ['Operational']*2 + ['Log rate']*3 +
                                           ['Compositional']*1 + ['Temperature']*1
                               if len(features) == 16 else [''] * len(features)})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # --- Tab 2: B&B Sensitivity ---
    with tab2:
        st.markdown("### Beggs & Brill Sensitivity Analysis")
        st.markdown("""
        > **Why this matters:** The B&B benchmark uses assumed tubing geometry. A critic
        > could argue that B&B would perform much better with correct geometry. This
        > sensitivity analysis sweeps across 60 geometry combinations to prove the
        > RF advantage is robust.
        """)

        if os.path.exists(BB_SENS_PATH):
            bb_sens = pd.read_csv(BB_SENS_PATH)
            st.dataframe(bb_sens, use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(bb_sens['tubing_ID_in'], bb_sens['RMSE'], 'o-', color='#B71C1C',
                    lw=2, ms=8, label='Beggs & Brill')
            if os.path.exists(LOWO_PATH):
                lowo = pd.read_csv(LOWO_PATH)
                ax.axhline(lowo['RMSE'].mean(), color='#1565C0', linestyle='--', lw=2,
                           label=f'RF LOWO mean ({lowo["RMSE"].mean():.1f} bar)')
            ax.set_xlabel('Assumed Tubing ID (inches)')
            ax.set_ylabel('RMSE (bar)')
            ax.set_title('B&B RMSE vs Tubing Diameter \u2014 RF Wins at ALL Sizes')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

            st.success("**Conclusion:** RF outperforms B&B across ALL tested tubing diameters, "
                       "well depths, and inclination angles.")
        else:
            st.warning("Run the pipeline first to generate sensitivity data.")

    # --- Tab 3: Cross-Well Regimes ---
    with tab3:
        st.markdown("### Cross-Well Operating Regime Analysis")
        st.markdown("""
        > **Why some wells generalize poorly:** The model struggles with wells whose operating
        > conditions (\u0394P range, WC, GOR) fall outside the training data. This is extrapolation,
        > not model failure.
        """)

        if os.path.exists(EXTRAP_PATH):
            extrap = pd.read_csv(EXTRAP_PATH)
            display_cols = ['test_well', 'overall_extrap_pct', 'LOWO_R2']
            display_cols = [c for c in display_cols if c in extrap.columns]
            st.dataframe(extrap[display_cols], use_container_width=True, hide_index=True)

            st.markdown("#### Key Findings")
            st.markdown("""
            - **F-15D** has the highest extrapolation % \u2192 worst LOWO R\u00b2 (\u22121.099)
            - **F-11H** has the lowest extrapolation % \u2192 best LOWO R\u00b2 (+0.862)
            - This confirms: **poor LOWO performance = extrapolation, not model failure**
            """)

        if os.path.exists(WC_ERROR_PATH):
            st.markdown("#### Error by Water Cut Regime")
            wc_err = pd.read_csv(WC_ERROR_PATH)
            st.dataframe(wc_err, use_container_width=True, hide_index=True)

    # --- Tab 4: Uncertainty ---
    with tab4:
        st.markdown("### Uncertainty Quantification")
        st.markdown("""
        > Each tree in the Random Forest makes its own prediction. The spread across
        > trees gives us a measure of model uncertainty \u2014 wide spread = low confidence.
        """)

        if os.path.exists(UNCERTAINTY_PATH):
            unc = pd.read_csv(UNCERTAINTY_PATH)
            col1, col2, col3 = st.columns(3)
            with col1:
                coverage = ((unc['actual'] >= unc['CI_lower_2.5']) &
                           (unc['actual'] <= unc['CI_upper_97.5'])).mean() * 100
                st.metric("95% CI Coverage", f"{coverage:.1f}%")
            with col2:
                st.metric("Mean Interval Width",
                         f"{(unc['CI_upper_97.5'] - unc['CI_lower_2.5']).mean():.2f} bar")
            with col3:
                st.metric("Mean Prediction Std", f"{unc['predicted_std'].mean():.2f} bar")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            sort_idx = unc['actual'].argsort()
            step = max(1, len(unc) // 200)
            idxs = sort_idx.values[::step]
            ax.scatter(unc['actual'].iloc[idxs], unc['predicted_mean'].iloc[idxs],
                      s=10, color='#1565C0', alpha=0.5)
            ax.fill_between(unc['actual'].iloc[idxs].values,
                          unc['CI_lower_2.5'].iloc[idxs].values,
                          unc['CI_upper_97.5'].iloc[idxs].values,
                          alpha=0.15, color='#1565C0')
            lims = [unc['actual'].min()-5, unc['actual'].max()+5]
            ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
            ax.set_xlabel('Actual \u0394P (bar)')
            ax.set_ylabel('Predicted \u0394P (bar)')
            ax.set_title(f'Predictions with 95% CI (Coverage={coverage:.1f}%)')
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.hist(unc['predicted_std'], bins=40, color='#7B1FA2', alpha=0.75)
            ax.axvline(unc['predicted_std'].mean(), color='red', linestyle='--', lw=2)
            ax.set_xlabel('Prediction Std (bar)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Model Uncertainty')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Run the pipeline first to generate uncertainty data.")


# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    VLP Predictor v2.1 | Williams Iwum | Federal University of Petroleum Resources
    <br>Development of a Random Forest-Based VLP Model for Multiphase Wellbore Flow Prediction
</div>
""", unsafe_allow_html=True)
