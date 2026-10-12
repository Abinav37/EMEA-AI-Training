#!/usr/bin/env python3
"""Train a beginner-friendly LSTM fraud detection model.

This project is intentionally simple and educational.

Important sequence note:
- The dataset contains transaction features but no customer IDs.
- We therefore create a time-ordered sequence window over recent rows,
  which is a defensible demonstration of sequential learning but not a true
  customer-history model.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
ARTIFACTS_DIR = ROOT / "artifacts"


def inspect_dataset(df: pd.DataFrame) -> None:
    print("\n=== Dataset inspection ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nClass distribution:")
    print(df["Class"].value_counts().to_string())
    print(f"Fraud rate: {df['Class'].mean():.6f}")
    print(f"Time min/max: {df['Time'].min()} / {df['Time'].max()}")
    print(f"Amount min/max: {df['Amount'].min()} / {df['Amount'].max()}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Sorted by Time: {df['Time'].is_monotonic_increasing}")


def choose_representative_subset(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Use a smaller, representative subset when the environment is limited."""
    if max_samples is None or len(df) <= max_samples:
        return df.copy()

    fraud = df[df["Class"] == 1].copy()
    normal = df[df["Class"] == 0].copy()

    n_fraud = min(len(fraud), max(50, max_samples // 10))
    n_normal = min(len(normal), max_samples - n_fraud)

    fraud_sample = fraud.sample(n=n_fraud, random_state=42)
    normal_sample = normal.sample(n=n_normal, random_state=42)
    subset = pd.concat([fraud_sample, normal_sample], ignore_index=True)
    subset = subset.sort_values("Time").reset_index(drop=True)

    print(f"\nUsing a representative subset of {len(subset)} rows for this run.")
    print(f"Fraud rows in subset: {len(subset[subset['Class'] == 1])}")
    print(f"Normal rows in subset: {len(subset[subset['Class'] == 0])}")
    return subset


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.8):
    train_end = int(len(df) * train_fraction)
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    test_df = df.iloc[train_end:].copy().reset_index(drop=True)
    return train_df, test_df


def create_sequences(frame: pd.DataFrame, sequence_length: int, feature_cols):
    values = frame[feature_cols].to_numpy(dtype=np.float32)
    labels = frame["Class"].to_numpy(dtype=np.float32)

    if len(frame) < sequence_length:
        raise ValueError(
            f"Not enough rows to create sequences of length {sequence_length}. "
            f"Received {len(frame)} rows."
        )

    num_windows = len(frame) - sequence_length + 1
    X = np.zeros((num_windows, sequence_length, len(feature_cols)), dtype=np.float32)
    y = np.zeros(num_windows, dtype=np.float32)

    for i in range(num_windows):
        window = values[i : i + sequence_length]
        X[i] = window
        y[i] = labels[i + sequence_length - 1]

    return X, y


def build_model(sequence_length: int, num_features: int):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, num_features)),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(model, scaler, metadata):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "fraud_lstm.keras"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    metadata_path = ARTIFACTS_DIR / "metadata.json"

    model.save(model_path)
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a small LSTM model for credit card fraud detection.")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    df = pd.read_csv(data_path)
    inspect_dataset(df)

    df = choose_representative_subset(df, args.max_samples)
    df = df.sort_values("Time").reset_index(drop=True)

    train_df, test_df = chronological_split(df, train_fraction=0.8)

    feature_cols = [col for col in df.columns if col not in ["Class"]]
    print(f"\nTraining rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Feature columns used: {feature_cols}")

    scaler = None
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    scaler = __import__("sklearn.preprocessing").preprocessing.StandardScaler()
    scaler.fit(train_df[feature_cols])
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    normal_train = train_scaled[train_scaled["Class"] == 0]
    fraud_train = train_scaled[train_scaled["Class"] == 1]

    target_ratio = 4.0
    target_normal_count = max(len(fraud_train), int(len(fraud_train) * target_ratio))
    if len(normal_train) > target_normal_count:
        normal_train = resample(
            normal_train,
            replace=False,
            n_samples=target_normal_count,
            random_state=42,
        )

    train_balanced = pd.concat([fraud_train, normal_train], ignore_index=True)
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nClass balance after training resampling:")
    print(train_balanced["Class"].value_counts().to_string())

    X_train, y_train = create_sequences(train_balanced, args.sequence_length, feature_cols)
    X_test, y_test = create_sequences(test_scaled, args.sequence_length, feature_cols)

    model = build_model(args.sequence_length, len(feature_cols))
    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        class_weight={0: 1.0, 1: 4.0},
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )

    y_prob = model.predict(X_test, verbose=0).ravel()
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Model evaluation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    metadata = {
        "sequence_length": args.sequence_length,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_columns": feature_cols,
        "threshold": threshold,
        "evaluation": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
        },
        "note": (
            "This model uses a time-ordered transaction window because the dataset contains no "
            "customer IDs or repeated customer histories. It is a demonstration model, not a real "
            "authorization system."
        ),
    }
    save_artifacts(model, scaler, metadata)


if __name__ == "__main__":
    main()
