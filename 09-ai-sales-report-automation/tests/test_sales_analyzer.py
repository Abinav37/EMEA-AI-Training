from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sales_analyzer import (
    clean_sales_data,
    compute_sales_summary,
    count_distinct_orders,
    validate_input_data,
)
from api.openrouter_client import generate_ai_report, normalize_report_markdown
from config.settings import settings


def test_settings_load_from_project_dotenv():
    assert settings.OPENROUTER_MODEL == "openrouter/free"
    assert isinstance(settings.OPENROUTER_API_KEY, str)
    assert bool(settings.OPENROUTER_API_KEY)


def test_report_markdown_is_normalized_for_rendering():
    report = "```markdown\n**Overall performance:**\n* Revenue was $1200.\n* Top product was Widget.\n```"
    normalized = normalize_report_markdown(report)
    assert "```" not in normalized
    assert "**Overall performance:**" in normalized
    assert "- Revenue was $1200." in normalized
    assert "- Top product was Widget." in normalized


def test_revenue_calculation_from_quantity_and_unit_price():
    df = pd.DataFrame(
        {
            "order_id": ["A1", "A1", "A2"],
            "product_name": ["Widget", "Widget", "Gadget"],
            "quantity": [2, 1, 3],
            "unit_price": [10, 10, 15],
            "revenue": [None, None, None],
            "region": ["North", "North", "South"],
        }
    )
    cleaned = clean_sales_data(df)
    assert cleaned["revenue"].isna().sum() == 0
    assert cleaned.loc[cleaned["order_id"] == "A1", "revenue"].sum() == 30
    assert cleaned.loc[cleaned["order_id"] == "A2", "revenue"].sum() == 45


def test_distinct_orders_are_counted_once_per_order_id():
    df = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O3"],
            "product_name": ["A", "A", "B", "C"],
            "quantity": [1, 1, 2, 1],
            "unit_price": [10, 10, 20, 5],
            "revenue": [10, 10, 40, 5],
            "region": ["North", "North", "West", "East"],
        }
    )
    cleaned = clean_sales_data(df)
    assert count_distinct_orders(cleaned) == 3


def test_duplicate_rows_are_dropped_and_missing_values_reported():
    df = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O3"],
            "product_name": ["A", "A", "B", "C"],
            "quantity": [1, 1, None, 2],
            "unit_price": [10, 10, 20, "bad"],
            "revenue": [10, 10, 40, 10],
            "region": ["North", "North", "West", "East"],
        }
    )
    cleaned = clean_sales_data(df)
    assert len(cleaned) == 1
    assert cleaned["quantity"].isna().sum() == 0
    assert cleaned["unit_price"].isna().sum() == 0
    assert cleaned["order_id"].tolist() == ["O1"]


def test_invalid_numeric_values_are_rejected_with_warnings():
    df = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3"],
            "product_name": ["A", "B", "C"],
            "quantity": [1, "invalid", 3],
            "unit_price": [10, 20, "oops"],
            "revenue": [10, 40, 60],
            "region": ["North", "South", "East"],
        }
    )
    cleaned, warnings = validate_input_data(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(warnings, list)
    assert any("invalid" in str(item).lower() for item in warnings)


def test_summary_has_expected_core_metrics():
    df = pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O3"],
            "product_name": ["A", "A", "B", "C"],
            "quantity": [2, 1, 3, 2],
            "unit_price": [10, 10, 5, 8],
            "revenue": [30, 30, 15, 16],
            "region": ["North", "North", "South", "North"],
        }
    )
    summary = compute_sales_summary(df)
    assert summary["total_revenue"] == 91
    assert summary["total_orders"] == 3
    assert summary["total_units_sold"] == 8
    assert summary["average_order_value"] == 30.333333333333332
