#!/usr/bin/env python3
"""Run a single banking-intent prediction with a saved DistilBERT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts" / "distilbert-banking-intent"
LABEL_MAP_PATH = ROOT / "artifacts" / "label_map.json"


def load_label_map() -> Dict[int, str]:
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            "The label map file was not found. Run training first: "
            "python 08-banking-intent-classification/train_intent_model.py"
        )

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        return {int(k): str(v) for k, v in json.load(file).items()}


def load_model_and_tokenizer():
    if not ARTIFACTS_DIR.exists() or not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            "The saved model artifacts were not found. Run training first: "
            "python 08-banking-intent-classification/train_intent_model.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACTS_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(ARTIFACTS_DIR))
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device, load_label_map()


def predict_with_loaded_model(
    query: str,
    model,
    tokenizer,
    label_map: Dict[int, str],
    device: torch.device,
) -> Tuple[str, float]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please enter a customer banking query before predicting.")

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0, pred_idx].item())

    label_name = label_map.get(pred_idx, str(pred_idx))
    return label_name, confidence


def predict_intent(query: str) -> Tuple[str, float]:
    model, tokenizer, device, label_map = load_model_and_tokenizer()
    return predict_with_loaded_model(query, model, tokenizer, label_map, device)


def main():
    parser = argparse.ArgumentParser(description="Predict banking intent for a customer message.")
    parser.add_argument("query", type=str, help="Customer text to classify.")
    args = parser.parse_args()

    try:
        label_name, confidence = predict_intent(args.query)
        print(f"Predicted intent: {label_name}")
        print(f"Confidence: {confidence:.4f}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
