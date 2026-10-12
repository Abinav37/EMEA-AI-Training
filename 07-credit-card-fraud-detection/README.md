# Credit Card Fraud Detection Demo

This project is a beginner-friendly demonstration of fraud detection using a time-ordered LSTM model on the Kaggle credit card fraud dataset.

Important notes:
- The dataset contains only transaction features and a `Class` label. It does not contain customer IDs or repeated customer histories.
- Because there are no customer identifiers, the sequence model is built from a time-ordered window of recent transaction rows rather than from true customer-session history.
- This is a teaching demo, not a production banking authorization or fraud-monitoring system.

## Project idea

We train a small LSTM model to predict whether a transaction is likely fraudulent. The workflow is kept intentionally simple:
1. Inspect the dataset
2. Sort transactions by time
3. Split chronologically into train and test sets
4. Scale features using training-only statistics
5. Build short time-sequence windows
6. Train a compact LSTM with class imbalance handling
7. Evaluate with precision, recall, F1-score, confusion matrix, and ROC-AUC
8. Save the model and scaler for a tiny Streamlit demo

## Repository layout

- `data/creditcard.csv` - raw dataset (kept out of Git)
- `train_lstm_fraud_model.py` - training script
- `app.py` - Streamlit demo app
- `requirements.txt` - Python dependencies
- `artifacts/` - saved model and preprocessing objects

## Data inspection summary

The dataset is expected to contain:
- `Time`
- `V1` to `V28` (PCA-transformed features)
- `Amount`
- `Class`

The raw dataset has 284,807 rows with 492 fraudulent transactions. That is a fraud rate of about 0.17%, so the dataset is extremely imbalanced.

## Setup

Create and activate a Python environment if needed, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r 07-credit-card-fraud-detection/requirements.txt
```

In the current environment, `streamlit` is missing, so the app cannot run until it is installed manually.

## Training

From the repository root:

```bash
python 07-credit-card-fraud-detection/train_lstm_fraud_model.py --max-samples 20000 --sequence-length 10 --epochs 6
```

The script uses a small representative subset by default for faster verification in limited environments. This is documented in the code and in the training output.

## Demo app

```bash
streamlit run 07-credit-card-fraud-detection/app.py
```

The app loads the saved model and scaler, lets the user adjust a fraud risk threshold, and shows the estimated fraud probability for a transaction.

This is intentionally a demo and should not be treated as an authorization decision system.

## Evaluation

The model is evaluated on the time-ordered test set using:
- precision
- recall
- F1-score
- confusion matrix
- ROC-AUC

## Notes on limitations

- We do not have customer IDs, so we cannot model a true customer-level fraud pattern.
- The sequence is only a rolling time window, not real customer behavior.
- This project is for learning and demonstration only.

## Git hygiene

The raw dataset and any virtual environment should remain out of Git. The repository root `.gitignore` includes the project CSV path and local virtual environments.
