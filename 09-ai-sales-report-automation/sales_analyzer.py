from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd


ORDER_DATE_ALIASES = {
    "order_date",
    "date",
    "sale_date",
    "order_date_time",
    "transaction_date",
    "invoice_date",
    "purchase_date",
}
ORDER_ID_ALIASES = {"order_id", "order", "order_number", "sales_order_id", "invoice_id", "id"}
PRODUCT_ALIASES = {"product_name", "product", "product_description", "item_name", "item", "item_name"}
QUANTITY_ALIASES = {"quantity", "qty", "units", "items", "count"}
UNIT_PRICE_ALIASES = {"unit_price", "price_per_unit", "unit_cost", "price", "sales_price"}
REVENUE_ALIASES = {"sales_amount", "revenue", "total_sales", "amount", "net_sales", "sales"}
REGION_ALIASES = {"region", "territory", "area", "market", "country"}


@dataclass
class SalesDataResult:
    dataframe: pd.DataFrame
    warnings: List[str]


def _normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _match_column(columns: List[str], candidates: set[str]) -> str | None:
    normalized_map = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]
    return None


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    new_columns = {}
    for column in renamed.columns:
        new_key = _normalize_column_name(column)
        new_columns[column] = new_key
    renamed = renamed.rename(columns=new_columns)
    return renamed


def coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().any():
        raise ValueError(f"Column '{column_name}' contains invalid numeric values.")
    return converted


def detect_or_create_aliases(df: pd.DataFrame) -> Dict[str, str]:
    renamed = _rename_columns(df)
    alias_map = {}
    alias_map["order_date"] = _match_column(list(renamed.columns), ORDER_DATE_ALIASES) or "order_date"
    alias_map["order_id"] = _match_column(list(renamed.columns), ORDER_ID_ALIASES) or "order_id"
    alias_map["product_name"] = _match_column(list(renamed.columns), PRODUCT_ALIASES) or "product_name"
    alias_map["quantity"] = _match_column(list(renamed.columns), QUANTITY_ALIASES) or "quantity"
    alias_map["unit_price"] = _match_column(list(renamed.columns), UNIT_PRICE_ALIASES) or "unit_price"
    alias_map["revenue"] = _match_column(list(renamed.columns), REVENUE_ALIASES) or "revenue"
    alias_map["region"] = _match_column(list(renamed.columns), REGION_ALIASES) or "region"
    return alias_map


def _standardize_data_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if df is None or df.empty:
        raise ValueError("The uploaded file is empty or contains no rows.")

    df = df.copy()
    df = _rename_columns(df)
    aliases = detect_or_create_aliases(df)

    required = ["order_id", "product_name", "quantity", "unit_price"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing required columns. Please include one of the expected names for: "
            + ", ".join(missing)
        )

    if aliases["order_date"] in df.columns:
        df["order_date"] = pd.to_datetime(df[aliases["order_date"]], errors="coerce")
        invalid_dates = df["order_date"].isna()
        if invalid_dates.any():
            warnings.append(f"Dropped {int(invalid_dates.sum())} rows with invalid dates.")
            df = df.loc[~invalid_dates].copy()
    else:
        df["order_date"] = pd.Timestamp("2000-01-01")
        warnings.append("No order date column was found; a placeholder date was used for summary calculations.")

    df["order_id"] = df[aliases["order_id"]].astype(str)
    df["product_name"] = df[aliases["product_name"]].fillna("Unknown").astype(str)
    df["quantity"] = pd.to_numeric(df[aliases["quantity"]], errors="coerce")
    df["unit_price"] = pd.to_numeric(df[aliases["unit_price"]], errors="coerce")

    quantity_invalid = df["quantity"].isna()
    unit_price_invalid = df["unit_price"].isna()
    invalid_numeric_rows = quantity_invalid | unit_price_invalid
    if invalid_numeric_rows.any():
        warnings.append(
            f"Dropped {int(invalid_numeric_rows.sum())} rows with missing or invalid quantity/unit_price values."
        )
        df = df.loc[~invalid_numeric_rows].copy()

    if aliases["revenue"] in df.columns:
        df["revenue"] = pd.to_numeric(df[aliases["revenue"]], errors="coerce")
        revenue_missing = df["revenue"].isna()
        if revenue_missing.any():
            warnings.append(f"{int(revenue_missing.sum())} revenue rows were missing or invalid and were recalculated when possible.")
        df.loc[revenue_missing, "revenue"] = df.loc[revenue_missing, "quantity"] * df.loc[revenue_missing, "unit_price"]
    else:
        df["revenue"] = df["quantity"] * df["unit_price"]
        warnings.append("Revenue was calculated from quantity × unit_price because no revenue column was available.")

    if aliases["region"] in df.columns:
        df["region"] = df[aliases["region"]].fillna("Unknown").astype(str)
    else:
        df["region"] = "Unknown"

    df = df.drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check the dataset for empty values or invalid entries.")

    return df, warnings


def validate_input_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if df is None:
        raise ValueError("No dataset was supplied.")
    df = df.copy()
    if df.empty:
        raise ValueError("The uploaded file is empty or contains no rows.")

    return _standardize_data_frame(df)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = validate_input_data(df)
    return cleaned


def count_distinct_orders(df: pd.DataFrame) -> int:
    if "order_id" in df.columns:
        return df["order_id"].nunique(dropna=True)
    return len(df)


def compute_sales_summary(df: pd.DataFrame) -> Dict[str, Any]:
    cleaned = clean_sales_data(df)

    revenue_col = cleaned["revenue"].sum()
    total_orders = count_distinct_orders(cleaned)
    total_units = float(cleaned["quantity"].sum()) if "quantity" in cleaned.columns else 0.0

    summary = {
        "total_revenue": float(revenue_col),
        "total_orders": int(total_orders),
        "total_units_sold": int(total_units),
        "average_order_value": float(revenue_col / total_orders) if total_orders else 0.0,
        "date_range": {
            "start": cleaned["order_date"].min().strftime("%Y-%m-%d") if not cleaned.empty else None,
            "end": cleaned["order_date"].max().strftime("%Y-%m-%d") if not cleaned.empty else None,
        },
        "revenue_by_month": (
            cleaned.assign(month=cleaned["order_date"].dt.to_period("M").astype(str))
            .groupby("month", as_index=False)["revenue"]
            .sum()
            .rename(columns={"month": "period", "revenue": "revenue"})
            .sort_values("period")
            .to_dict(orient="records")
        ),
        "top_products": (
            cleaned.groupby("product_name", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(10)
            .rename(columns={"product_name": "product", "revenue": "revenue"})
            .to_dict(orient="records")
        ),
        "regional_revenue": (
            cleaned.groupby("region", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .rename(columns={"region": "region", "revenue": "revenue"})
            .to_dict(orient="records")
        ),
        "data_quality_warnings": [],
    }
    return summary
