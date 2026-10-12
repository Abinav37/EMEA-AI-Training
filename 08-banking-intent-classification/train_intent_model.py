#!/usr/bin/env python3
"""Train a DistilBERT classifier for banking customer intent detection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_NAME = "distilbert-base-uncased"


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path)
    required = {"text", "label", "label_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    return df


def inspect_dataset(name: str, df: pd.DataFrame) -> None:
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {df.isnull().sum().to_dict()}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Unique intents: {df['label_text'].nunique()}")
    print("Class distribution:")
    print(df["label_text"].value_counts().to_string())


def validate_label_mapping(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame):
    train_map = dict(train_df.groupby("label")["label_text"].first())
    dev_map = dict(dev_df.groupby("label")["label_text"].first())
    test_map = dict(test_df.groupby("label")["label_text"].first())

    if train_map != dev_map or train_map != test_map:
        raise ValueError(
            "Label mapping differs across train/dev/test sets. "
            f"Train={train_map}\nDev={dev_map}\nTest={test_map}"
        )

    label_to_name = {int(k): str(v) for k, v in train_map.items()}
    return label_to_name


def build_dataloader(df: pd.DataFrame, tokenizer: AutoTokenizer, batch_size: int, shuffle: bool, max_length: int = 64):
    dataset = IntentDataset(
        texts=df["text"].fillna("").astype(str).tolist(),
        labels=df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model, tokenizer, df: pd.DataFrame, label_to_name, batch_size: int, max_length: int = 64):
    model.eval()
    all_preds = []
    all_labels = []

    loader = build_dataloader(df, tokenizer, batch_size=batch_size, shuffle=False, max_length=max_length)
    with torch.no_grad():
        for batch in loader:
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[label_to_name[i] for i in sorted(label_to_name)],
        digits=4,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a small DistilBERT intent classifier on banking texts.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs. Keep this small for a MacBook Air.")
    parser.add_argument("--max-train-samples", type=int, default=50, help="Representative sample size for a Mac-friendly run.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--max-length", type=int, default=64, help="Tokenizer max length used to keep the CPU run lightweight.")
    args = parser.parse_args()

    train_df = load_split("train.csv")
    dev_df = load_split("dev.csv")
    test_df = load_split("test.csv")

    label_to_name = validate_label_mapping(train_df, dev_df, test_df)

    if args.max_train_samples is not None and len(train_df) > args.max_train_samples:
        num_classes = train_df["label"].nunique()
        n_per_class = max(1, args.max_train_samples // num_classes)
        sampled_frames = []
        for _, group in train_df.groupby("label", sort=False):
            n_take = min(len(group), n_per_class)
            sampled_frames.append(group.sample(n=n_take, random_state=42))
        train_df = pd.concat(sampled_frames, ignore_index=True)
        print(f"\nUsing a representative training subset of {len(train_df)} rows ({n_per_class} per class) for this run.")

    for name, df in [("train.csv", train_df), ("dev.csv", dev_df), ("test.csv", test_df)]:
        inspect_dataset(name, df)

    print("\nValidated label mapping:")
    print({k: v for k, v in list(label_to_name.items())[:5]})
    print(f"Number of classes: {len(label_to_name)}")
    print("Note: This is a lightweight run using a representative subset of training data for faster Mac-friendly validation.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_to_name),
        id2label={idx: label_name for idx, label_name in label_to_name.items()},
        label2id={label_name: idx for idx, label_name in label_to_name.items()},
    )

    train_loader = build_dataloader(train_df, tokenizer, batch_size=args.batch_size, shuffle=True, max_length=args.max_length)
    dev_loader = build_dataloader(dev_df, tokenizer, batch_size=args.batch_size, shuffle=False, max_length=args.max_length)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}: average training loss = {avg_loss:.4f}")

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs, labels=labels)
                dev_loss += outputs.loss.item()
        print(f"Dev loss = {dev_loss / max(len(dev_loader), 1):.4f}")
        model.train()

    test_metrics = evaluate_model(model, tokenizer, test_df, label_to_name, batch_size=args.batch_size, max_length=args.max_length)
    print("\n=== Test evaluation ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro precision: {test_metrics['macro_precision']:.4f}")
    print(f"Macro recall: {test_metrics['macro_recall']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print("\nClassification report:\n", test_metrics["classification_report"])
    print("\nConfusion matrix:\n", test_metrics["confusion_matrix"])

    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(ARTIFACTS_DIR / "distilbert-banking-intent")
    tokenizer.save_pretrained(ARTIFACTS_DIR / "distilbert-banking-intent")

    with open(ARTIFACTS_DIR / "label_map.json", "w", encoding="utf-8") as file:
        json.dump({str(k): v for k, v in label_to_name.items()}, file, indent=2)

    print(f"\nSaved artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
