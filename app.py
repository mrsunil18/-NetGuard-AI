from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "intrusion_model.joblib"
FEATURES_PATH = ROOT / "models" / "feature_columns.json"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
IMPORTANCE_PATH = ROOT / "outputs" / "feature_importance.csv"

st.set_page_config(page_title="NetGuard AI | Intrusion Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ NetGuard AI")
st.subheader("Machine Learning Network Intrusion Detection")
st.write(
    "Classifies network-flow records as **BENIGN** or **ATTACK** using a trained "
    "Random Forest model. This is an educational research prototype, not a live firewall."
)

if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.markdown("1. Download CIC-IDS2017 **MachineLearningCSV.zip** from the official dataset page.")
    st.markdown("2. Extract its CSV files into `data/raw/`.")
    st.markdown("3. Run `python train.py` in your terminal.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Attack precision", f"{metrics['precision_attack'] * 100:.2f}%")
    c3.metric("Attack recall", f"{metrics['recall_attack'] * 100:.2f}%")
    c4.metric("Attack F1-score", f"{metrics['f1_attack'] * 100:.2f}%")

st.divider()
st.header("Analyze a CSV file")
st.write("Upload a CIC-IDS2017-style flow CSV. The file should contain the same feature columns used during training.")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded, low_memory=False)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        original = df.copy()

        # Drop labels from incoming data; predictions must not use the answer column.
        df = df.drop(columns=[c for c in df.columns if c.lower().strip() in {"label", "class", "target", "attack"}], errors="ignore")

        # Align uploaded features to the training columns.
        for feature in features:
            if feature not in df.columns:
                df[feature] = np.nan
        X = df[features].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        with st.spinner("Analyzing network flows..."):
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

        results = original.copy()
        results["Prediction"] = np.where(predictions == 1, "ATTACK", "BENIGN")
        results["Attack probability (%)"] = (probabilities * 100).round(2)

        total = len(results)
        attacks = int((predictions == 1).sum())
        a, b, c = st.columns(3)
        a.metric("Rows analyzed", f"{total:,}")
        b.metric("Predicted attacks", f"{attacks:,}")
        c.metric("Predicted benign", f"{total - attacks:,}")

        st.dataframe(results.head(500), use_container_width=True)
        st.caption("Showing up to 500 rows in the preview.")
        st.download_button(
            "Download all predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="intrusion_predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
        st.info("Use CSVs with numeric network-flow features similar to the CIC-IDS2017 MachineLearningCSV files.")

st.divider()
if IMPORTANCE_PATH.exists():
    st.header("Most important model features")
    importance = pd.read_csv(IMPORTANCE_PATH)
    st.bar_chart(importance.set_index("feature")["importance"])
    st.caption("Feature importance shows what the fitted model used; it does not prove a feature causes an attack.")

st.info(
    "Safety note: results can be wrong. Do not use this prototype alone to block traffic, "
    "discipline users, or make operational security decisions."
)
