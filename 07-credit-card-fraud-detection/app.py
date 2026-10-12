import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_lstm.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, scaler, metadata


st.set_page_config(page_title="Fraud Demo")
st.title("Credit Card Fraud Detection Demo")
st.caption("Demonstration only. This app is not a real banking authorization system.")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "No trained model artifacts were found. Run the training script first: "
        "python 07-credit-card-fraud-detection/train_lstm_fraud_model.py"
    )
    st.stop()

model, scaler, metadata = load_model_and_scaler()
feature_columns = metadata["feature_columns"]
sequence_length = metadata["sequence_length"]

def build_input_sequence(values):
    values = np.asarray(values, dtype=np.float32)
    sequence = np.tile(values, (sequence_length, 1))
    sequence = sequence.reshape(1, sequence_length, len(values))
    return sequence

st.subheader("Transaction input")

transaction = {}
for col in feature_columns:
    if col == "Time":
        default = 36000.0
    elif col == "Amount":
        default = 50.0
    else:
        default = 0.0
    transaction[col] = st.number_input(col, value=float(default), step=0.01)

user_df = pd.DataFrame([transaction], columns=feature_columns)
scaled_values = scaler.transform(user_df[feature_columns])
sequence = build_input_sequence(scaled_values[0])
probability = float(model.predict(sequence, verbose=0)[0][0])

threshold = st.slider("Risk threshold", min_value=0.0, max_value=1.0, value=metadata["threshold"], step=0.01)
label = "Fraud likely" if probability >= threshold else "Low risk"

st.metric("Fraud probability", f"{probability:.3f}")
st.metric("Decision", label)

st.markdown(
    """
    This demo uses a short time-ordered sequence window because the dataset
    contains no customer IDs or customer-level history. It is meant for teaching,
    not real financial authorization.
    """
)
