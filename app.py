from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# Prevent joblib/sklearn threading issues on macOS
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from src.data.schema import HEART_SCHEMA
from src.models.inference import predict_from_frame, load_feature_columns
from src.models.persistence import load_model


@st.cache_resource
def load_cached_model(model_path: str | Path):
    """Cache model loading to prevent issues with Streamlit reruns."""
    return load_model(model_path)


# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("❤️ Heart Disease Prediction Model")
st.markdown("---")

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")

    model_path = Path("models/random_forest_model.joblib")
    metrics_path = Path("models/random_forest_metrics.json")
    features_path = Path("models/feature_columns.txt")

    if model_path.exists() and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")

        with st.expander("Confusion Matrix"):
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual No Disease", "Actual Disease"],
                    columns=["Predicted No Disease", "Predicted Disease"],
                )
                st.write(cm_df)
    else:
        st.warning("Model files not found. Please train the model first.")

# Main content
tab1, tab2 = st.tabs(["📊 Single Prediction", "📈 Batch Prediction"])

with tab1:
    st.header("Make a Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=400, value=200)
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=202, value=150)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

    with col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    col3, col4 = st.columns(2)
    with col3:
        slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    with col4:
        ca = st.number_input("Major Vessels (ca)", min_value=0, max_value=4, value=0)
        thal = st.number_input("Thalassemia (thal)", min_value=0, max_value=3, value=2)

    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        if model_path.exists() and features_path.exists():
            try:
                input_data = pd.DataFrame(
                    {
                        "age": [age],
                        "sex": [sex],
                        "cp": [cp],
                        "trestbps": [trestbps],
                        "chol": [chol],
                        "fbs": [fbs],
                        "restecg": [restecg],
                        "thalach": [thalach],
                        "exang": [exang],
                        "oldpeak": [oldpeak],
                        "slope": [slope],
                        "ca": [ca],
                        "thal": [thal],
                    }
                )

                # Load model once with caching to prevent threading issues
                cached_model = load_cached_model(model_path)
                prediction = predict_from_frame(
                    model=cached_model,
                    feature_frame=input_data,
                    feature_columns=features_path,
                )[0]

                st.markdown("---")
                if prediction == 1:
                    st.warning(f"⚠️ **High Risk**: The model predicts **DISEASE PRESENT** (Heart Disease likelihood)")
                else:
                    st.success(f"✅ **Low Risk**: The model predicts **NO DISEASE** (Normal)")

                with st.expander("📋 Patient Summary"):
                    summary_df = pd.DataFrame(
                        {
                            "Feature": list(input_data.columns),
                            "Value": list(input_data.values[0]),
                        }
                    )
                    st.dataframe(summary_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("Model files not found. Please train the model first.")


with tab2:
    st.header("Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file with features", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)

            if "target" in batch_data.columns:
                batch_data = batch_data.drop(columns=["target"])

            if st.button("🔮 Predict Batch", use_container_width=True, type="primary"):
                if model_path.exists() and features_path.exists():
                    # Load model once with caching to prevent threading issues
                    cached_model = load_cached_model(model_path)
                    predictions = predict_from_frame(
                        model=cached_model,
                        feature_frame=batch_data,
                        feature_columns=features_path,
                    )

                    result_df = batch_data.copy()
                    result_df["prediction"] = predictions
                    result_df["risk_level"] = result_df["prediction"].apply(lambda x: "🔴 Disease" if x == 1 else "🟢 No Disease")

                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Model files not found. Please train the model first.")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
