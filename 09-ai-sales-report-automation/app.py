from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from sales_analyzer import compute_sales_summary, clean_sales_data, validate_input_data
from report_generator import generate_ai_report


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"


def _supported_file_filter() -> str:
    return ".csv, .xlsx, .xls"


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Supported types are CSV and Excel (.xlsx/.xls).")


def _get_available_data_files() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}],
        key=lambda p: p.name.lower(),
    )


def _build_sample_csv() -> str:
    sample = pd.DataFrame(
        {
            "order_id": ["1001", "1002", "1003"],
            "order_date": ["2024-01-05", "2024-01-08", "2024-01-10"],
            "product_name": ["Laptop", "Mouse", "Keyboard"],
            "quantity": [2, 5, 3],
            "unit_price": [800, 25, 60],
            "region": ["North", "South", "East"],
        }
    )
    sample["revenue"] = sample["quantity"] * sample["unit_price"]
    return sample.to_csv(index=False)


def _render_kpis(summary: dict[str, Any]) -> None:
    metrics = [
        ("Total revenue", summary.get("total_revenue", 0.0), "currency"),
        ("Total orders", summary.get("total_orders", 0), "number"),
        ("Units sold", summary.get("total_units_sold", 0), "number"),
        ("Average order value", summary.get("average_order_value", 0.0), "currency"),
    ]

    cards = st.columns(len(metrics))
    for index, (label, value, kind) in enumerate(metrics):
        with cards[index]:
            if kind == "currency":
                st.metric(label, f"${float(value):,.2f}")
            else:
                st.metric(label, f"{int(value):,}")


def _render_chart_maybe(summary: dict[str, Any], kind: str) -> None:
    if kind == "revenue_by_month":
        records = summary.get("revenue_by_month", [])
        if not records:
            return
        df = pd.DataFrame(records)
        st.line_chart(df.set_index("period")["revenue"])
    elif kind == "top_products":
        records = summary.get("top_products", [])
        if not records:
            return
        df = pd.DataFrame(records)
        st.bar_chart(df.set_index("product")["revenue"])
    elif kind == "regional_revenue":
        records = summary.get("regional_revenue", [])
        if not records:
            return
        df = pd.DataFrame(records)
        st.bar_chart(df.set_index("region")["revenue"])


def main() -> None:
    st.set_page_config(page_title="AI Sales Report Automation", page_icon="📊", layout="wide")

    st.title("AI Sales Report Automation")
    st.caption("Upload a sales file or select a dataset in the project data folder to generate verified sales metrics and an AI summary.")

    dataset_files = _get_available_data_files()
    file_selection = st.selectbox(
        "Select a local dataset in the data folder",
        options=[path.name for path in dataset_files] + ["Upload a new file"],
        index=0 if dataset_files else len(["Upload a new file"]) - 1,
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel sales file",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV, XLSX, and XLS files. The file is processed in the browser and then analyzed locally.",
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
    elif file_selection == "Upload a new file":
        file_name = None
    else:
        file_name = file_selection

    st.markdown("---")
    if st.button("Analyze Sales", type="primary"):
        try:
            if uploaded_file is not None:
                source = uploaded_file
                df = pd.read_csv(source) if source.name.lower().endswith(".csv") else pd.read_excel(source)
            elif file_selection and file_selection != "Upload a new file":
                selected_path = DATA_DIR / file_selection
                df = _read_file(selected_path)
            else:
                st.warning("Please choose a dataset or upload a CSV/Excel file before analyzing.")
                st.stop()

            cleaned, warnings = validate_input_data(df)
            summary = compute_sales_summary(cleaned)
            summary["data_quality_warnings"] = warnings

            st.session_state["summary"] = summary
            st.session_state["data"] = cleaned
            st.session_state["warnings"] = warnings
            st.session_state["loaded_file"] = file_name or "uploaded file"
        except Exception as exc:  # pragma: no cover - UI path for user clarity
            st.error(f"Could not analyze the file: {exc}")
            st.stop()

    if "summary" in st.session_state and "data" in st.session_state:
        summary = st.session_state["summary"]
        cleaned = st.session_state["data"]
        warnings = st.session_state.get("warnings", [])

        _render_kpis(summary)

        st.markdown("### Revenue over time")
        _render_chart_maybe(summary, "revenue_by_month")

        st.markdown("### Top products")
        _render_chart_maybe(summary, "top_products")

        st.markdown("### Regional revenue")
        _render_chart_maybe(summary, "regional_revenue")

        st.markdown("### Data quality and assumptions")
        for warning in warnings:
            st.warning(warning)

        if "revenue" in cleaned.columns and "quantity" in cleaned.columns and "unit_price" in cleaned.columns:
            st.info("Revenue is calculated from quantity × unit_price when a revenue column is missing. Existing revenue values are preserved to avoid double-counting.")

        st.markdown("### AI-generated executive report")
        ai_report = generate_ai_report(summary)
        st.markdown(ai_report)

        st.markdown("### Downloadable output")
        report_text = "\n".join([
            "# AI Sales Report",
            "",
            f"Generated for: {st.session_state.get('loaded_file', 'uploaded file')}",
            "",
            f"- Total revenue: ${summary.get('total_revenue', 0.0):,.2f}",
            f"- Total orders: {summary.get('total_orders', 0):,}",
            f"- Total units sold: {summary.get('total_units_sold', 0):,}",
            f"- Average order value: ${summary.get('average_order_value', 0.0):,.2f}",
            "",
            ai_report,
        ])
        st.download_button("Download report (.md)", report_text, file_name="sales_report.md", mime="text/markdown")

        summary_df = pd.DataFrame(summary.get("top_products", []))
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download summary CSV", csv_bytes, file_name="sales_summary.csv", mime="text/csv")

        st.markdown("### Raw verified summary")
        st.json(summary)

    else:
        st.markdown("### Sample template")
        st.download_button(
            label="Download sample CSV",
            data=_build_sample_csv(),
            file_name="sales_sample.csv",
            mime="text/csv",
        )
        st.info("Upload a CSV or Excel file or use the sample template to start. The app will calculate only supported metrics from the actual columns present.")


if __name__ == "__main__":
    main()
