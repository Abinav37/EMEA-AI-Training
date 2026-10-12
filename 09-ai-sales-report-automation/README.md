# AI Sales Report Automation

This project provides a beginner-friendly sales analytics dashboard with a locally calculated summary and optional AI executive summary through OpenRouter.

## Overview

The app reads sales data from CSV or Excel files, validates the schema and data quality, calculates reliable sales metrics in Pandas, and optionally generates an executive summary through LangChain + OpenRouter when a valid API key is configured.

## Folder structure

```text
09-ai-sales-report-automation/
├── app.py
├── sales_analyzer.py
├── report_generator.py
├── requirements.txt
├── README.md
├── .env
├── .env.example
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── tests/
│   └── test_sales_analyzer.py
└── .venv/  # repo-level virtual environment (not tracked)
```

## Setup

From the repository root:

```bash
cd /Users/abinav/EMEA-AI-Training
./.venv/bin/python -m pip install -r 09-ai-sales-report-automation/requirements.txt
```

## Configuration

Create a local real `.env` file in this project root:

```bash
cd /Users/abinav/EMEA-AI-Training/09-ai-sales-report-automation
cp .env.example .env
```

Then fill in your actual values:

```env
OPENROUTER_API_KEY=your_actual_api_key_here
OPENROUTER_MODEL=your_model_id_here
```

The app reads these values from `config/settings.py` via `python-dotenv`.

Important:
- The real `.env` file is not tracked by Git.
- `.env.example` is safe to commit and contains placeholders only.
- The API key is never printed, exposed in the UI, or logged.

## Supported file formats

The app supports:
- CSV files
- Excel `.xlsx`
- Excel `.xls` when the required reader dependency is available

It also accepts uploaded CSV or Excel files directly in the Streamlit app.

## Expected columns

The analyzer accepts common naming variations for:
- Order date
- Order ID
- Product name
- Quantity
- Unit price
- Sales amount or revenue
- Region

If the dataset uses different names, rename columns before analysis or map them in your local data file. The app does not assume a single fixed schema.

## Data quality handling

The project intentionally does not silently replace invalid values with zeros.

Rules:
- Empty files are rejected.
- Invalid dates are removed with a warning.
- Missing or invalid numeric values are removed with a warning.
- Duplicate rows are removed.
- Revenue is calculated as quantity × unit_price when a revenue column is missing.
- If a revenue column exists, it is preferred and no double-counting is performed.
- Order totals are counted by distinct order ID, not by repeated product rows.

## Metrics

The app calculates metrics only when the column data supports them, including:
- Total revenue
- Total orders
- Total units sold
- Average order value
- Revenue by month
- Top products
- Regional revenue
- Date range
- Data quality warnings

No period-over-period change is computed unless the dataset supports a fair comparison.

## AI report generation

The optional AI report is generated in `report_generator.py` using LangChain + OpenRouter.

Behavior:
- The app passes verified metrics to the LLM, not the raw sales data.
- The AI layer explains only the values supplied.
- It does not invent figures or unsupported claims.
- If the API key is missing or the model request fails, the app still shows the verified metrics and explains that AI report generation is unavailable.

## Launch the app

From the repository root:

```bash
cd /Users/abinav/EMEA-AI-Training
./.venv/bin/python -m streamlit run 09-ai-sales-report-automation/app.py --server.headless true --server.port 8503
```

Then open:

```text
http://localhost:8503
```

## Run tests

```bash
cd /Users/abinav/EMEA-AI-Training
./.venv/bin/python -m pytest 09-ai-sales-report-automation/tests/test_sales_analyzer.py
```

## Troubleshooting

- If the model cannot load or the OpenRouter key is missing, the app will still calculate the sales metrics.
- If the file format is not supported, the app explains the file type issue.
- If the data is incomplete or invalid, the app shows clear warnings rather than silent data loss.
- If the API fails due to network, rate-limit, or model availability, the UI clearly states that AI report generation is unavailable.

## Limitations

- This is a beginner-friendly demo project, not a production analytics platform.
- OpenRouter access depends on a valid API key and model availability from the external provider.
- Local data quality issues are surfaced as warnings and do not result in fabricated numbers.
