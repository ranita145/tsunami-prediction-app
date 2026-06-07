from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import (
    DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_dataframe,
    train_and_save,
)


st.set_page_config(
    page_title="Tsunami Potential Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 232, 255, 0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(8, 145, 178, 0.15), transparent 30%),
        linear-gradient(135deg, #041016 0%, #071A22 45%, #02070A 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06141C 0%, #0B2530 100%);
    border-right: 1px solid rgba(82, 224, 255, 0.20);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #F3FBFF !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

.hero-card {
    position: relative;
    padding: 34px 36px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(10, 42, 55, 0.94), rgba(5, 20, 28, 0.96)),
        url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80');
    background-blend-mode: overlay;
    background-size: cover;
    border: 1px solid rgba(83, 235, 255, 0.22);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(3, 13, 18, 0.85), rgba(3, 13, 18, 0.55), rgba(3, 13, 18, 0.78));
    z-index: 0;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(56, 232, 255, 0.13);
    color: #9AF3FF;
    border: 1px solid rgba(56, 232, 255, 0.28);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .02em;
    margin-bottom: 16px;
}

.hero-title {
    font-size: clamp(34px, 5vw, 64px);
    line-height: 1.02;
    font-weight: 800;
    letter-spacing: -0.05em;
    color: #FFFFFF;
    margin: 0;
    max-width: 920px;
}

.hero-subtitle {
    margin-top: 16px;
    color: #B6D8E2;
    font-size: 17px;
    max-width: 820px;
}

.hero-warning {
    margin-top: 24px;
    padding: 15px 18px;
    border-radius: 16px;
    background: rgba(255, 209, 102, 0.13);
    border: 1px solid rgba(255, 209, 102, 0.26);
    color: #FFE7A3;
    font-weight: 600;
}

.section-card {
    padding: 24px;
    border-radius: 24px;
    background: rgba(8, 30, 40, 0.74);
    border: 1px solid rgba(123, 226, 255, 0.16);
    box-shadow: 0 18px 55px rgba(0, 0, 0, 0.20);
    margin-top: 18px;
}

.mini-card {
    padding: 18px 20px;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(16, 49, 63, 0.90), rgba(7, 24, 33, 0.94));
    border: 1px solid rgba(114, 231, 255, 0.14);
    min-height: 118px;
}

.mini-label {
    color: #9CCBD7;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 8px;
}

.mini-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.04em;
}

.mini-note {
    color: #A9C6CE;
    font-size: 13px;
    margin-top: 5px;
}

.result-card {
    padding: 26px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(14, 54, 70, 0.95), rgba(5, 21, 29, 0.96));
    border: 1px solid rgba(56, 232, 255, 0.25);
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
    margin-top: 20px;
}

.result-title {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-desc {
    color: #B8DDE7;
    font-size: 15px;
    margin-bottom: 16px;
}

.high-risk {
    color: #FF6B6B;
    font-weight: 800;
}

.medium-risk {
    color: #FFD166;
    font-weight: 800;
}

.low-risk {
    color: #58F29C;
    font-weight: 800;
}

div[data-testid="stMetric"] {
    background: rgba(7, 28, 37, 0.78);
    border: 1px solid rgba(123, 226, 255, 0.16);
    padding: 18px 20px;
    border-radius: 20px;
}

div[data-testid="stMetricLabel"] p {
    color: #9ECBD6 !important;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid rgba(143, 227, 255, 0.15);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(8, 30, 40, 0.65);
    border-radius: 999px;
    color: #D7F8FF;
    padding: 10px 18px;
    border: 1px solid rgba(143, 227, 255, 0.12);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1BB8D4, #28E8FF) !important;
    color: #021014 !important;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 14px !important;
    border: 0 !important;
    background: linear-gradient(135deg, #1BB8D4, #38E8FF) !important;
    color: #031017 !important;
    font-weight: 800 !important;
    padding: 0.75rem 1.2rem !important;
    box-shadow: 0 12px 28px rgba(56, 232, 255, 0.22);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    background-color: rgba(4, 18, 25, 0.85) !important;
    border: 1px solid rgba(129, 224, 255, 0.18) !important;
    color: #FFFFFF !important;
}

hr {
    border-color: rgba(143, 227, 255, 0.12);
}

.footer {
    margin-top: 32px;
    color: #7EA8B5;
    font-size: 13px;
    text-align: center;
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_or_train_model():
    """Load saved sklearn pipeline. If missing, train it automatically."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_and_save(DATA_PATH, MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return bundle, metadata


def get_tsunami_probability(model, data: pd.DataFrame) -> np.ndarray:
    """Return probability for class 1 = tsunami if predict_proba is available."""
    if not hasattr(model, "predict_proba"):
        return np.full(shape=(len(data),), fill_value=np.nan)

    probabilities = model.predict_proba(data)
    classes = list(model.named_steps["classifier"].classes_)

    if 1 in classes:
        class_index = classes.index(1)
    elif "1" in classes:
        class_index = classes.index("1")
    else:
        class_index = 1 if probabilities.shape[1] > 1 else 0

    return probabilities[:, class_index]


def probability_to_risk_label(probability: float) -> str:
    if pd.isna(probability):
        return "Unknown"
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def risk_class(risk_label: str) -> str:
    if risk_label == "High":
        return "high-risk"
    if risk_label == "Medium":
        return "medium-risk"
    if risk_label == "Low":
        return "low-risk"
    return ""


def make_prediction(input_df: pd.DataFrame, model):
    data = prepare_dataframe(input_df)
    data = data[FEATURES]
    prediction = model.predict(data)
    probability = get_tsunami_probability(model, data)

    result = input_df.copy()
    result["predicted_tsunami"] = prediction.astype(int)
    result["probability_tsunami"] = probability
    result["risk_label"] = [probability_to_risk_label(p) for p in probability]

    return result


bundle, metadata = load_or_train_model()
model = bundle["model"]
metrics = metadata.get("holdout_metrics", {})

# Sidebar
with st.sidebar:
    st.markdown("## 🌊 Model Center")
    st.markdown("**Selected model:**  \nTuned Random Forest")
    st.markdown("**Target:** `tsunami`")
    st.markdown("---")

    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    st.metric("Recall Macro", f"{metrics.get('recall_macro', 0):.4f}")
    st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")

    st.markdown("---")
    st.markdown("### Class Label")
    st.markdown("**0** = Non-tsunami")
    st.markdown("**1** = Tsunami-related")
    st.markdown("---")
    st.caption("Academic prototype only. Not an official tsunami warning system.")

# Hero
st.markdown(
    """
    <div class="hero-card">
      <div class="hero-content">
        <div class="badge">🌐 IS411 Data Modelling • Group 09</div>
        <h1 class="hero-title">Tsunami Potential Prediction</h1>
        <p class="hero-subtitle">
          A machine learning dashboard to classify tsunami potential based on earthquake characteristics
          such as magnitude, depth, intensity, location, alert category, and seismic indicators.
        </p>
        <div class="hero-warning">
          ⚠️ Academic prototype only. This dashboard is not a replacement for official tsunami early warning systems.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric cards below hero
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Main Model</div>
          <div class="mini-value">Random Forest</div>
          <div class="mini-note">Tuned with selected hyperparameters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Holdout F1 Macro</div>
          <div class="mini-value">{metrics.get('f1_macro', 0):.4f}</div>
          <div class="mini-note">Primary metric for imbalanced target</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    class_dist = metadata.get("class_distribution", {})
    total_records = sum(class_dist.values()) if isinstance(class_dist, dict) else 0
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Dataset Records</div>
          <div class="mini-value">{total_records}</div>
          <div class="mini-note">Earthquake events used for modelling</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

tab_single, tab_batch, tab_about = st.tabs([
    "🔍 Single Prediction",
    "📁 Batch CSV Prediction",
    "📌 About Deployment",
])

# Load dataset only for default values and selectbox options.
if DATA_PATH.exists():
    reference_df = pd.read_csv(DATA_PATH)
else:
    reference_df = pd.DataFrame()


def get_category_options(column, fallback):
    if not reference_df.empty and column in reference_df.columns:
        values = reference_df[column].dropna().astype(str).sort_values().unique().tolist()
        return ["Unknown"] + values[:120]
    return fallback


with tab_single:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Single Earthquake Prediction")
    st.write("Input earthquake characteristics below, then click **Predict tsunami potential**.")

    st.markdown("#### 🌋 Core Earthquake Characteristics")
    col1, col2, col3 = st.columns(3)

    with col1:
        magnitude = st.number_input("Magnitude", value=6.80, min_value=0.0, max_value=10.0, step=0.1)
        depth = st.number_input("Depth (km)", value=25.0, min_value=0.0, max_value=800.0, step=1.0)
        latitude = st.number_input("Latitude", value=-4.9559, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")

    with col2:
        cdi = st.number_input("CDI", value=5, min_value=0, max_value=12, step=1)
        mmi = st.number_input("MMI", value=4, min_value=0, max_value=12, step=1)
        longitude = st.number_input("Longitude", value=100.7380, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")

    with col3:
        sig = st.number_input("Significance score (sig)", value=735, min_value=0, step=1)
        nst = st.number_input("Number of seismic stations (nst)", value=99, min_value=0, step=1)
        gap = st.number_input("Azimuthal gap", value=34.0, min_value=0.0, max_value=360.0, step=1.0)

    st.markdown("#### 🛰️ Seismic & Location Metadata")
    col4, col5, col6 = st.columns(3)
    with col4:
        dmin = st.number_input("Minimum distance (dmin)", value=2.229, min_value=0.0, step=0.001, format="%.3f")
        alert = st.selectbox("Alert", get_category_options("alert", ["Unknown", "green", "yellow", "orange", "red"]))
    with col5:
        mag_type = st.selectbox("Magnitude Type", get_category_options("magType", ["Unknown", "mww", "mw", "mb", "ms"]))
        net = st.selectbox("Seismic Network Code", get_category_options("net", ["Unknown", "us"]))
    with col6:
        continent = st.selectbox("Continent", get_category_options("continent", ["Unknown", "Asia", "Oceania", "North America", "South America", "Europe", "Africa"]))
        country = st.text_input("Country", value="Indonesia")

    single_input = pd.DataFrame([{
        "magnitude": magnitude,
        "cdi": cdi,
        "mmi": mmi,
        "sig": sig,
        "nst": nst,
        "dmin": dmin,
        "gap": gap,
        "depth": depth,
        "latitude": latitude,
        "longitude": longitude,
        "alert": None if alert == "Unknown" else alert,
        "net": None if net == "Unknown" else net,
        "magType": None if mag_type == "Unknown" else mag_type,
        "continent": None if continent == "Unknown" else continent,
        "country": None if country.strip() == "" else country,
    }])

    predict_clicked = st.button("Predict tsunami potential", type="primary", use_container_width=True)

    if predict_clicked:
        result = make_prediction(single_input, model)
        pred = int(result.loc[0, "predicted_tsunami"])
        prob = float(result.loc[0, "probability_tsunami"])
        risk = result.loc[0, "risk_label"]
        css_class = risk_class(risk)

        label = "Tsunami-related" if pred == 1 else "Non-tsunami"
        emoji = "🌊" if pred == 1 else "✅"

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-title">{emoji} Prediction Result: {label}</div>
              <div class="result-desc">
                The model estimates a tsunami-related probability of <b>{prob:.2%}</b>.
                Risk category: <span class="{css_class}">{risk}</span>.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Class", f"{pred}")
        res_col2.metric("Probability Tsunami", f"{prob:.2%}")
        res_col3.metric("Risk Label", risk)

        st.markdown("#### Prediction Detail")
        st.dataframe(
            result[["predicted_tsunami", "probability_tsunami", "risk_label"] + FEATURES],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


with tab_batch:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV file containing earthquake records. The app will generate predicted class, tsunami probability, and risk label.")

    with st.expander("Required feature columns"):
        st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader("Upload earthquake CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("#### Uploaded Data Preview")
        st.dataframe(batch_df.head(10), use_container_width=True, hide_index=True)

        result_df = make_prediction(batch_df, model)

        st.markdown("#### Prediction Result")
        display_columns = ["predicted_tsunami", "probability_tsunami", "risk_label"] + [col for col in FEATURES if col in result_df.columns]
        st.dataframe(result_df[display_columns], use_container_width=True, hide_index=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction result as CSV",
            data=csv_bytes,
            file_name="tsunami_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


with tab_about:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("About This Deployment")
    st.markdown(
        """
        This dashboard deploys the selected **Tuned Random Forest** model as an academic prototype.

        **Workflow:**
        1. The app loads the saved `tsunami_tuned_random_forest.joblib` model bundle.
        2. New data is processed using the same preprocessing pipeline used during training.
        3. The pipeline applies median imputation, categorical imputation, one-hot encoding, scaling, and model prediction.
        4. The output contains predicted class, probability score, and simple risk category.

        **Important limitation:**
        Some variables such as `alert` and `sig` may be post-event attributes. For realistic early-warning use, a separate model should be trained using only early-available earthquake features.
        """
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown("#### Model Metadata")
        st.json({
            "project_title": metadata.get("project_title"),
            "model_name": metadata.get("model_name"),
            "target": metadata.get("target"),
            "dataset_rows": metadata.get("dataset_rows"),
            "dataset_columns": metadata.get("dataset_columns"),
        })
    with col_meta2:
        st.markdown("#### Holdout Metrics")
        st.json(metrics)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="footer">
      IS411 Data Modelling • Group 09 • Tsunami Potential Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
