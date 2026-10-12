# Banking Intent Classification with DistilBERT

This project classifies customer banking questions into one of many banking intents using a pretrained DistilBERT model.

## Purpose

The goal is to build a beginner-friendly text-classification example that uses:
- real training, development, and test CSV files
- a pretrained transformer model
- proper validation and evaluation
- a simple inference script and a Streamlit demo for new queries

This project is for learning and demonstration only. It is not a production banking system.

## Dataset

The raw data is stored in:
- `data/train.csv`
- `data/dev.csv`
- `data/test.csv`

The real schema was inspected before training. The files contain:
- `text`: the customer message
- `label`: numeric class id
- `label_text`: intent label string

## Environment setup

From the repository root:

```bash
cd /Users/abinav/EMEA-AI-Training
./.venv/bin/python -m pip install -r 08-banking-intent-classification/requirements.txt
```

The current environment uses a CPU-first setup for reliability on a MacBook Air. The saved model artifacts are already present under the project folder and are used for real inference.

## UI demo

Start the Streamlit app from the repository root:

```bash
cd /Users/abinav/EMEA-AI-Training
./.venv/bin/python -m streamlit run 08-banking-intent-classification/app.py --server.headless true --server.port 8501
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

The app includes:
- a large customer-query text area
- a few example banking queries
- a predicted intent card with confidence
- a confidence warning for uncertain predictions
- a short workflow explanation
- a compact summary of verified dataset and evaluation values

## Training

If you want to regenerate the artifacts on a more capable machine, run:

```bash
python 08-banking-intent-classification/train_intent_model.py --epochs 1 --max-train-samples 500 --batch-size 8
```

This project intentionally uses a smaller representative training subset for a MacBook Air instead of claiming full-dataset results. The script still uses the real train/dev/test split:
1. validates the CSV schema
2. prints dataset diagnostics
3. trains on a representative training subset
4. validates on the development split
5. evaluates the final model on the held-out test split
6. saves model artifacts under `artifacts/`

## Inference

```bash
python 08-banking-intent-classification/predict_intent.py "I cannot find my card and it has not arrived yet"
```

The script prints:
- predicted intent label
- confidence score

## Verified project values

The actual dataset sizes are verified from the CSV files:
- train rows: 10,003
- dev rows: 3,080
- test rows: 3,080
- classes: 77

The actual evaluation metrics verified from the training logs for the lightweight demo run are:
- accuracy: 0.0162
- macro precision: 0.0008
- macro recall: 0.0162
- macro F1: 0.0014

These are demo-run metrics only and should not be interpreted as a production-quality benchmark.

## Artifacts

Training writes the model, tokenizer, and label mapping to:

```text
08-banking-intent-classification/artifacts/
```

These are intentionally kept out of Git because generated model files can be large.

## Notes

- The dataset includes 77 classes.
- The development and test splits are balanced in class count.
- The model uses a safe text preprocessing approach: keep the original wording, strip empty values, and tokenize cleanly.
- No customer IDs or database logic are used.
- Confidence is the model's estimate only; it is not a guarantee of correctness.

## Git hygiene

The repository root `.gitignore` keeps raw CSV files, virtual environments, caches, and generated model artifacts out of Git.
