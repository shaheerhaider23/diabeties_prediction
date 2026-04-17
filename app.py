import warnings
from pathlib import Path
from typing import Dict, Tuple
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Diabetes Prediction Studio",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, #9fd8cb 0%, transparent 55%),
                radial-gradient(900px 400px at 95% 0%, #f9d8a3 0%, transparent 50%),
                linear-gradient(180deg, #f8f5ee 0%, #f1f7f6 100%);
            color: #1e2a2f;
            font-family: 'Manrope', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: 0.2px; 
            color: #12343b;
        }

        .hero {
            background: linear-gradient(110deg, #12343b 0%, #1f5460 60%, #2d6a78 100%);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            color: #f5faf8;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(18, 52, 59, 0.2);
        }

        .hero p {
            margin: 0.2rem 0;
            opacity: 0.95;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid #d3e3df;
            border-radius: 14px;
            padding: 0.8rem;
        }

        .stButton > button {
            background: linear-gradient(90deg, #12343b 0%, #2d6a78 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            transition: transform 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            background: linear-gradient(90deg, #0f2c32 0%, #255762 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

REPLACE_ZERO_WITH_NAN = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "diabetes.csv"
MODELS_DIR = BASE_DIR / "notebook" / "models"

NORMAL_RANGES = {
    "Pregnancies": (0, 3),
    "Glucose": (70, 99),
    "BloodPressure": (60, 80),
    "SkinThickness": (10, 30),
    "Insulin": (16, 166),
    "BMI": (18.5, 24.9),
    "DiabetesPedigreeFunction": (0.08, 0.5),
    "Age": (18, 45),
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_pretrained_models() -> Tuple[Dict[str, object], StandardScaler]:
    """Load pre-trained models from pickle files"""
    try:
        with open(MODELS_DIR / "logistic_model.pkl", "rb") as f:
            logistic_model = pickle.load(f)
        with open(MODELS_DIR / "random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open(MODELS_DIR / "svm_model.pkl", "rb") as f:
            svm_model = pickle.load(f)
        with open(MODELS_DIR / "knn_model.pkl", "rb") as f:
            knn_model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        models = {
            "Logistic Regression": logistic_model,
            "Random Forest": rf_model,
            "Support Vector Machine": svm_model,
            "K-Nearest Neighbors": knn_model,
        }
        
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure models are trained and saved: {e}")
        st.stop()


@st.cache_resource
def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare training and test data"""
    data = load_data().copy()
    data[REPLACE_ZERO_WITH_NAN] = data[REPLACE_ZERO_WITH_NAN].replace(0, np.nan)
    data.fillna(data.median(), inplace=True)

    X = data[FEATURES]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, y_train, X_test, y_test


def evaluate_models(
    models: Dict[str, object], 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    scaler: StandardScaler
) -> pd.DataFrame:
    """Evaluate pre-trained models on test set"""
    rows = []
    for name, model in models.items():
        # Scale data for models that need it
        if name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled)
        else:  # Random Forest
            pred = model.predict(X_test)
        
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 Score": f1_score(y_test, pred, zero_division=0),
            }
        )

    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    return results.reset_index(drop=True)


def input_panel() -> pd.DataFrame:
    with st.sidebar:
        st.header("Patient Inputs")

        st.markdown("### Normal Range Guide")
        st.caption("Values above normal ranges generally increase diabetes risk likelihood.")

        pregnancies = st.slider("Pregnancies", 0, 20, 2)
        st.caption("Normal: 0 - 3")

        glucose = st.slider("Glucose", 0, 220, 120)
        st.caption("Normal fasting: 70 - 99 mg/dL")

        blood_pressure = st.slider("BloodPressure", 0, 140, 72)
        st.caption("Normal diastolic: 60 - 80 mmHg")

        skin_thickness = st.slider("SkinThickness", 0, 100, 20)
        st.caption("Typical: 10 - 30 mm")

        insulin = st.slider("Insulin", 0, 900, 80)
        st.caption("Typical fasting: 16 - 166 uIU/mL")

        bmi = st.slider("BMI", 0.0, 70.0, 32.0, 0.1)
        st.caption("Healthy: 18.5 - 24.9")

        dpf = st.slider("DiabetesPedigreeFunction", 0.05, 2.5, 0.47, 0.01)
        st.caption("Lower is generally better: 0.08 - 0.50")

        age = st.slider("Age", 18, 90, 33)
        st.caption("Lower risk baseline: 18 - 45")

        inputs = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        high_risk_features = [
            feature
            for feature, value in inputs.items()
            if value > NORMAL_RANGES[feature][1]
        ]

        if high_risk_features:
            st.warning(
                "Higher than normal: "
                + ", ".join(high_risk_features)
                + ". Risk is more likely to increase."
            )
        else:
            st.success("Most selected values are within the shown normal ranges.")

    patient = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
            }
        ]
    )
    return patient


def build_metrics_chart(model_results: pd.DataFrame):
    metrics_df = model_results.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )
    metrics_df["Score"] = (metrics_df["Score"] * 100).round(2)

    return (
        alt.Chart(metrics_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", sort="-y", title=None),
            y=alt.Y("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Metric:N", legend=alt.Legend(orient="top")),
            tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2f")],
            xOffset="Metric:N",
        )
        .properties(height=360)
    )


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin-bottom:0.2rem;">Diabetes Prediction Studio</h2>
            <p>Live prediction with trained models from notebook.</p>
            <p>Dataset: PIMA Indians Diabetes | Target: Outcome (0/1)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load pre-trained models and scaler
    models, scaler = load_pretrained_models()
    _, _, X_test, y_test = prepare_data()
    
    # Evaluate models
    model_results = evaluate_models(models, X_test, y_test, scaler)

    left, right = st.columns([1.05, 1.15])
    with left:
        st.subheader("Model Performance")
        display_df = model_results.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            display_df[col] = (display_df[col] * 100).round(2)
        st.dataframe(display_df, use_container_width=True)

    with right:
        st.subheader("Performance Comparison")
        st.altair_chart(build_metrics_chart(model_results), use_container_width=True)

    st.markdown("---")

    st.subheader("Live Prediction")
    selected_model_name = st.selectbox("Choose model", list(models.keys()))
    selected_model = models[selected_model_name]

    patient_data = input_panel()

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.write("Input preview")
        st.dataframe(patient_data, use_container_width=True)

    with c2:
        if st.button("Predict Diabetes Risk"):
            # Scale data if needed
            if selected_model_name in ["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"]:
                patient_scaled = scaler.transform(patient_data)
                pred = int(selected_model.predict(patient_scaled)[0])
                proba = None
                if hasattr(selected_model, "predict_proba"):
                    proba = float(selected_model.predict_proba(patient_scaled)[0][1])
            else:  # Random Forest
                pred = int(selected_model.predict(patient_data)[0])
                proba = None
                if hasattr(selected_model, "predict_proba"):
                    proba = float(selected_model.predict_proba(patient_data)[0][1])

            if pred == 1:
                st.error("Prediction: Diabetic (1)")
            else:
                st.success("Prediction: Non-Diabetic (0)")

            if proba is not None:
                st.metric("Diabetes Probability", f"{proba * 100:.2f}%")



if __name__ == "__main__":
    main()
