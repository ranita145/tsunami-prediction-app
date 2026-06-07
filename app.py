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

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1220px;
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
    background: linear-gradient(90deg, rgba(3, 13, 18, 0.88), rgba(3, 13, 18, 0.56), rgba(3, 13, 18, 0.78));
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
    font-weight: 800;
    letter-spacing: .02em;
    margin-bottom: 16px;
}

.hero-title {
    font-size: clamp(34px, 5vw, 62px);
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
    max-width: 860px;
}

.hero-warning {
    margin-top: 24px;
    padding: 15px 18px;
    border-radius: 16px;
    background: rgba(255, 209, 102, 0.13);
    border: 1px solid rgba(255, 209, 102, 0.26);
    color: #FFE7A3;
    font-weight: 700;
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
    font-weight: 800;
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

.high-risk { color: #FF6B6B; font-weight: 800; }
.medium-risk { color: #FFD166; font-weight: 800; }
.low-risk { color: #58F29C; font-weight: 800; }

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

.interactive-note {
    padding: 13px 16px;
    border-radius: 16px;
    background: rgba(56, 232, 255, 0.10);
    border: 1px solid rgba(56, 232, 255, 0.18);
    color: #C8F8FF;
    font-size: 14px;
    margin-bottom: 18px;
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
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_and_save(DATA_PATH, MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return bundle, metadata


def get_tsunami_probability(model, data: pd.DataFrame) -> np.ndarray:
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


def build_single_input_from_state():
    return pd.DataFrame([{
        "magnitude": st.session_state.get("magnitude", 6.8),
        "cdi": st.session_state.get("cdi", 5),
        "mmi": st.session_state.get("mmi", 4),
        "sig": st.session_state.get("sig", 735),
        "nst": st.session_state.get("nst", 99),
        "dmin": st.session_state.get("dmin", 2.229),
        "gap": st.session_state.get("gap", 34.0),
        "depth": st.session_state.get("depth", 25.0),
        "latitude": st.session_state.get("latitude", -4.9559),
        "longitude": st.session_state.get("longitude", 100.7380),
        "alert": None if st.session_state.get("alert", "Unknown") == "Unknown" else st.session_state.get("alert"),
        "net": None if st.session_state.get("net", "Unknown") == "Unknown" else st.session_state.get("net"),
        "magType": None if st.session_state.get("mag_type", "Unknown") == "Unknown" else st.session_state.get("mag_type"),
        "continent": None if st.session_state.get("continent", "Unknown") == "Unknown" else st.session_state.get("continent"),
        "country": None if str(st.session_state.get("country", "")).strip() == "" else st.session_state.get("country"),
    }])


def set_scenario(scenario):
    for key, value in scenario.items():
        st.session_state[key] = value


SCENARIOS = {
    "Indonesia Offshore Strong": {
        "magnitude": 7.8, "depth": 18.0, "latitude": -4.9559, "longitude": 100.7380,
        "cdi": 7, "mmi": 7, "sig": 1100, "nst": 140, "dmin": 1.2, "gap": 28.0,
        "alert": "red", "mag_type": "mww", "net": "us", "continent": "Asia", "country": "Indonesia",
    },
    "Japan Moderate": {
        "magnitude": 6.7, "depth": 35.0, "latitude": 38.2970, "longitude": 142.3720,
        "cdi": 5, "mmi": 5, "sig": 700, "nst": 120, "dmin": 1.8, "gap": 38.0,
        "alert": "yellow", "mag_type": "mww", "net": "us", "continent": "Asia", "country": "Japan",
    },
    "Deep Inland Low Risk": {
        "magnitude": 5.8, "depth": 250.0, "latitude": 35.0, "longitude": 70.0,
        "cdi": 3, "mmi": 3, "sig": 420, "nst": 60, "dmin": 4.5, "gap": 80.0,
        "alert": "green", "mag_type": "mb", "net": "us", "continent": "Asia", "country": "Afghanistan",
    },
}


bundle, metadata = load_or_train_model()
model = bundle["model"]
metrics = metadata.get("holdout_metrics", {})

if DATA_PATH.exists():
    reference_df = pd.read_csv(DATA_PATH)
else:
    reference_df = pd.DataFrame()


def get_category_options(column, fallback):
    if not reference_df.empty and column in reference_df.columns:
        values = reference_df[column].dropna().astype(str).sort_values().unique().tolist()
        return ["Unknown"] + values[:120]
    return fallback


# Initialize session defaults
DEFAULTS = {
    "magnitude": 6.8,
    "depth": 25.0,
    "latitude": -4.9559,
    "longitude": 100.7380,
    "cdi": 5,
    "mmi": 3,
    "sig": 737,
    "nst": 99,
    "dmin": 2.229,
    "gap": 30.0,
    "alert": "red",
    "mag_type": "mb",
    "net": "duputel",
    "continent": "Asia",
    "country": "Indonesia",
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


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


st.markdown(
    """
    <div class="hero-card">
      <div class="hero-content">
        <div class="badge">🌐 IS411 Data Modelling • Group 09</div>
        <h1 class="hero-title">Interactive Tsunami Potential Dashboard</h1>
        <p class="hero-subtitle">
          Explore earthquake scenarios, adjust parameters, compare what-if results, and upload CSV data
          to classify tsunami potential using the selected Tuned Random Forest model.
        </p>
        <div class="hero-warning">
          ⚠️ Academic prototype only. This dashboard is not a replacement for official tsunami early warning systems.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""<div class="mini-card"><div class="mini-label">Main Model</div><div class="mini-value">Random Forest</div><div class="mini-note">Tuned with selected hyperparameters</div></div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""<div class="mini-card"><div class="mini-label">Holdout F1 Macro</div><div class="mini-value">{metrics.get('f1_macro', 0):.4f}</div><div class="mini-note">Primary metric for imbalanced target</div></div>""", unsafe_allow_html=True)
with col_c:
    class_dist = metadata.get("class_distribution", {})
    total_records = sum(class_dist.values()) if isinstance(class_dist, dict) else 0
    st.markdown(f"""<div class="mini-card"><div class="mini-label">Dataset Records</div><div class="mini-value">{total_records}</div><div class="mini-note">Earthquake events used for modelling</div></div>""", unsafe_allow_html=True)

st.markdown("")

tab_single, tab_simulator, tab_map, tab_batch, tab_about = st.tabs([
    "🔍 Interactive Prediction",
    "🧪 What-if Simulator",
    "🗺️ Location Explorer",
    "📁 Batch CSV Prediction",
    "📌 About Deployment",
])


with tab_single:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Interactive Earthquake Prediction")
    st.markdown('<div class="interactive-note">Choose a preset scenario or manually adjust earthquake parameters. The prediction updates when you click the button.</div>', unsafe_allow_html=True)

    st.markdown("#### Quick Scenario Presets")
    p1, p2, p3 = st.columns(3)
    if p1.button("🌊 Indonesia Offshore Strong", use_container_width=True):
        set_scenario(SCENARIOS["Indonesia Offshore Strong"])
        st.rerun()
    if p2.button("🇯🇵 Japan Moderate", use_container_width=True):
        set_scenario(SCENARIOS["Japan Moderate"])
        st.rerun()
    if p3.button("✅ Deep Inland Low Risk", use_container_width=True):
        set_scenario(SCENARIOS["Deep Inland Low Risk"])
        st.rerun()

    st.markdown("#### 🌋 Core Earthquake Characteristics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider("Magnitude", 0.0, 10.0, key="magnitude", step=0.1)
        st.slider("Depth (km)", 0.0, 800.0, key="depth", step=1.0)
        st.number_input("Latitude", min_value=-90.0, max_value=90.0, key="latitude", step=0.0001, format="%.4f")
    with col2:
        st.slider("CDI", 0, 12, key="cdi", step=1)
        st.slider("MMI", 0, 12, key="mmi", step=1)
        st.number_input("Longitude", min_value=-180.0, max_value=180.0, key="longitude", step=0.0001, format="%.4f")
    with col3:
        st.slider("Significance score (sig)", 0, 3000, key="sig", step=1)
        st.slider("Number of seismic stations (nst)", 0, 1000, key="nst", step=1)
        st.slider("Azimuthal gap", 0.0, 360.0, key="gap", step=1.0)

    st.markdown("#### 🛰️ Seismic & Location Metadata")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.number_input("Minimum distance (dmin)", min_value=0.0, key="dmin", step=0.001, format="%.3f")
        alert_options = get_category_options("alert", ["Unknown", "green", "yellow", "orange", "red"])
        if st.session_state["alert"] not in alert_options:
            alert_options.append(st.session_state["alert"])
        st.selectbox("Alert", alert_options, key="alert")
    with col5:
        mag_options = get_category_options("magType", ["Unknown", "mww", "mw", "mb", "ms"])
        if st.session_state["mag_type"] not in mag_options:
            mag_options.append(st.session_state["mag_type"])
        st.selectbox("Magnitude Type", mag_options, key="mag_type")

        net_options = get_category_options("net", ["Unknown", "us"])
        if st.session_state["net"] not in net_options:
            net_options.append(st.session_state["net"])
        st.selectbox("Seismic Network Code", net_options, key="net")
    with col6:
        continent_options = get_category_options("continent", ["Unknown", "Asia", "Oceania", "North America", "South America", "Europe", "Africa"])
        if st.session_state["continent"] not in continent_options:
            continent_options.append(st.session_state["continent"])
        st.selectbox("Continent", continent_options, key="continent")
        st.text_input("Country", key="country")

    current_input = build_single_input_from_state()

    if st.button("Predict tsunami potential", type="primary", use_container_width=True):
        result = make_prediction(current_input, model)
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
                Estimated tsunami-related probability: <b>{prob:.2%}</b>.
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
        st.progress(min(max(prob, 0), 1), text=f"Tsunami probability: {prob:.2%}")

        st.markdown("#### Current Location")
        try:
            map_df = pd.DataFrame({"lat": [float(st.session_state["latitude"])], "lon": [float(st.session_state["longitude"])]})
            st.map(map_df, latitude="lat", longitude="lon", zoom=3)
        except Exception:
            st.info("Map preview is unavailable for the current coordinate input.")

        st.markdown("#### Prediction Detail")
        st.dataframe(result[["predicted_tsunami", "probability_tsunami", "risk_label"] + FEATURES], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


with tab_simulator:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What-if Simulator")
    st.write("This section compares the current input with automatic variations. It helps demonstrate how changes in magnitude, depth, and alert category may affect the predicted probability.")

    base_input = build_single_input_from_state()
    scenarios = []

    def add_variant(name, changes):
        row = base_input.copy()
        for k, v in changes.items():
            row[k] = v
        row["scenario"] = name
        scenarios.append(row)

    add_variant("Current input", {})
    add_variant("Magnitude +0.5", {"magnitude": min(float(base_input.loc[0, "magnitude"]) + 0.5, 10.0)})
    add_variant("Magnitude +1.0", {"magnitude": min(float(base_input.loc[0, "magnitude"]) + 1.0, 10.0)})
    add_variant("Shallower depth", {"depth": max(float(base_input.loc[0, "depth"]) - 20.0, 0.0)})
    add_variant("Deeper depth", {"depth": min(float(base_input.loc[0, "depth"]) + 100.0, 800.0)})
    add_variant("Alert green", {"alert": "green"})
    add_variant("Alert red", {"alert": "red"})

    scenario_df = pd.concat(scenarios, ignore_index=True)
    scenario_names = scenario_df.pop("scenario")
    scenario_result = make_prediction(scenario_df, model)
    scenario_result.insert(0, "scenario", scenario_names)

    chart_df = scenario_result[["scenario", "probability_tsunami"]].set_index("scenario")
    st.bar_chart(chart_df)

    st.dataframe(
        scenario_result[["scenario", "predicted_tsunami", "probability_tsunami", "risk_label", "magnitude", "depth", "alert", "latitude", "longitude"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Interpretation tip:** This is not causal explanation. It is an interactive sensitivity check based on model predictions.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


with tab_map:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Location Explorer")
    st.write("Move the coordinate input in the first tab, then open this tab to preview the selected epicenter location.")

    loc_col1, loc_col2 = st.columns([1, 2])
    with loc_col1:
        current_input = build_single_input_from_state()
        loc_result = make_prediction(current_input, model)
        loc_prob = float(loc_result.loc[0, "probability_tsunami"])
        loc_pred = int(loc_result.loc[0, "predicted_tsunami"])
        loc_risk = loc_result.loc[0, "risk_label"]
        st.metric("Current Latitude", f"{float(st.session_state['latitude']):.4f}")
        st.metric("Current Longitude", f"{float(st.session_state['longitude']):.4f}")
        st.metric("Predicted Class", f"{loc_pred}")
        st.metric("Tsunami Probability", f"{loc_prob:.2%}")
        st.metric("Risk Label", loc_risk)

    with loc_col2:
        try:
            map_df = pd.DataFrame({
                "lat": [float(st.session_state["latitude"])],
                "lon": [float(st.session_state["longitude"])],
            })
            st.map(map_df, latitude="lat", longitude="lon", zoom=3)
        except Exception:
            st.info("Map preview is unavailable for the current coordinate input.")

    st.markdown("#### Dataset Geospatial Preview")
    if not reference_df.empty and {"latitude", "longitude"}.issubset(reference_df.columns):
        sample_size = min(500, len(reference_df))
        sample_df = reference_df[["latitude", "longitude"]].dropna().sample(sample_size, random_state=42)
        sample_df = sample_df.rename(columns={"latitude": "lat", "longitude": "lon"})
        st.map(sample_df, latitude="lat", longitude="lon", zoom=1)
    else:
        st.info("Dataset geospatial preview is unavailable.")
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

        st.markdown("#### Prediction Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Total Records", len(result_df))
        summary_col2.metric("Predicted Tsunami", int((result_df["predicted_tsunami"] == 1).sum()))
        summary_col3.metric("Average Probability", f"{result_df['probability_tsunami'].mean():.2%}")

        st.bar_chart(result_df["risk_label"].value_counts())

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

        **Interactive components added:**
        - Quick scenario presets
        - Adjustable sliders for earthquake characteristics
        - Single prediction with probability and risk label
        - What-if simulator
        - Location map preview
        - Batch CSV upload with prediction summary

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
      IS411 Data Modelling • Group 09 • Interactive Tsunami Potential Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)
