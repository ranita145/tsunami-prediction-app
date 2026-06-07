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
)


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

st.title("🌊 Tsunami Potential Prediction Based on Earthquake Characteristics")
st.caption("Academic prototype for IS411 Data Modelling — not an official tsunami early warning system.")

st.warning(
    "Disclaimer: This application is only for academic analysis and decision support. "
    "It must not be used as a replacement for official tsunami early warning systems."
)

with st.sidebar:
    st.header("Model Information")
    st.write("**Selected model:** Tuned Random Forest")
    st.write("**Target:** `tsunami`")
    st.write("**Class 0:** Non-tsunami")
    st.write("**Class 1:** Tsunami-related")

    metrics = metadata.get("holdout_metrics", {})
    st.metric("Holdout Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    st.metric("Holdout Recall Macro", f"{metrics.get('recall_macro', 0):.4f}")
    st.metric("Holdout F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")

    st.write("**Dataset class distribution:**")
    st.json(metadata.get("class_distribution", {}))

tab_single, tab_batch, tab_about = st.tabs([
    "Single Prediction",
    "Batch CSV Prediction",
    "About Deployment",
])

# Load dataset only for default values and selectbox options.
if DATA_PATH.exists():
    reference_df = pd.read_csv(DATA_PATH)
else:
    reference_df = pd.DataFrame()


def get_category_options(column, fallback):
    if not reference_df.empty and column in reference_df.columns:
        values = reference_df[column].dropna().astype(str).sort_values().unique().tolist()
        return ["Unknown"] + values[:100]
    return fallback


with tab_single:
    st.subheader("Single Earthquake Prediction")
    st.write("Enter earthquake characteristics, then click **Predict tsunami potential**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        magnitude = st.number_input("Magnitude", value=7.0, min_value=0.0, max_value=10.0, step=0.1)
        depth = st.number_input("Depth (km)", value=25.0, min_value=0.0, max_value=800.0, step=1.0)
        latitude = st.number_input("Latitude", value=-4.9559, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitude", value=100.7380, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")

    with col2:
        cdi = st.number_input("CDI", value=4, min_value=0, max_value=12, step=1)
        mmi = st.number_input("MMI", value=4, min_value=0, max_value=12, step=1)
        sig = st.number_input("Significance score (sig)", value=735, min_value=0, step=1)
        nst = st.number_input("Number of seismic stations (nst)", value=99, min_value=0, step=1)

    with col3:
        dmin = st.number_input("Minimum distance (dmin)", value=2.229, min_value=0.0, step=0.001, format="%.3f")
        gap = st.number_input("Azimuthal gap", value=34.0, min_value=0.0, max_value=360.0, step=1.0)
        alert = st.selectbox("Alert", get_category_options("alert", ["Unknown", "green", "yellow", "orange", "red"]))
        mag_type = st.selectbox("Magnitude Type", get_category_options("magType", ["Unknown", "mww", "mw", "mb", "ms"]))

    col4, col5 = st.columns(2)
    with col4:
        net = st.selectbox("Seismic Network Code", get_category_options("net", ["Unknown", "us"]))
    with col5:
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

    if st.button("Predict tsunami potential", type="primary"):
        result = make_prediction(single_input, model)
        pred = int(result.loc[0, "predicted_tsunami"])
        prob = float(result.loc[0, "probability_tsunami"])
        risk = result.loc[0, "risk_label"]

        left, middle, right = st.columns(3)
        left.metric("Predicted class", f"{pred} - {'Tsunami' if pred == 1 else 'Non-tsunami'}")
        middle.metric("Probability tsunami", f"{prob:.2%}")
        right.metric("Risk label", risk)

        st.dataframe(result[["predicted_tsunami", "probability_tsunami", "risk_label"] + FEATURES], use_container_width=True)

with tab_batch:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV file containing earthquake records. The app will use the required feature columns below.")
    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader("Upload earthquake CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview uploaded data:")
        st.dataframe(batch_df.head(), use_container_width=True)

        result_df = make_prediction(batch_df, model)

        st.write("Prediction result:")
        display_columns = ["predicted_tsunami", "probability_tsunami", "risk_label"] + [col for col in FEATURES if col in result_df.columns]
        st.dataframe(result_df[display_columns], use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction result as CSV",
            data=csv_bytes,
            file_name="tsunami_prediction_result.csv",
            mime="text/csv",
        )

with tab_about:
    st.subheader("About This Deployment")
    st.markdown(
        """
        This Streamlit app deploys the selected **Tuned Random Forest** model as a safe academic prototype.

        **How it works:**
        1. The app loads `tsunami_tuned_random_forest.joblib`.
        2. If the model file does not exist, the app trains the model from `earthquake_data.csv`.
        3. The same preprocessing pipeline is applied during training and prediction:
           median imputation, most-frequent categorical imputation, one-hot encoding, scaling, and Random Forest prediction.
        4. The output includes predicted class, tsunami probability, and a simple risk label.

        **Important limitation:**
        Some variables such as `alert` and `sig` may be post-event attributes. For real early-warning use, a separate model should be trained using only early-available earthquake features.
        """
    )

    st.json(metadata)
