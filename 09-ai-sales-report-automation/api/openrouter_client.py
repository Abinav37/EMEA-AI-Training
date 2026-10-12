from __future__ import annotations

import re
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import settings


def normalize_report_markdown(report_text: str) -> str:
    text = (report_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text

    text = re.sub(r"^```(?:markdown)?\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^```(?:markdown)?\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?m)^\s*\*\s+", "- ", text)
    return text.strip()


def build_report_prompt(summary: Dict[str, Any]) -> PromptTemplate:
    template = """
You are preparing an executive summary based only on the supplied sales facts.

Instructions:
- Use the exact metrics provided below.
- Do not invent or guess any numbers.
- If a metric is missing, say it was not available.
- Do not claim causal effects or business outcomes that are not supported by the data.
- Mention uncertainty clearly when the dataset is limited.

Verified sales summary:
{summary}

Write a brief, professional executive summary with:
1. One paragraph on overall performance.
2. One paragraph on top products and regions.
3. One paragraph on data quality and limitations.
4. A closing note saying the summary is based strictly on the provided data.
"""
    return PromptTemplate(input_variables=["summary"], template=template)


def generate_ai_report(summary: Dict[str, Any]) -> str:
    if not settings.OPENROUTER_API_KEY:
        return "AI report generation is unavailable because OPENROUTER_API_KEY is not configured. Sales metrics are still calculated and displayed below."

    try:
        llm = ChatOpenAI(
            model=settings.OPENROUTER_MODEL,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
        prompt = build_report_prompt(summary)
        chain = prompt | llm
        response = chain.invoke({"summary": str(summary)})
        response_text = response.content if hasattr(response, "content") else str(response)
        return normalize_report_markdown(response_text)
    except Exception as exc:  # pragma: no cover - intentionally graceful
        return f"AI report generation is unavailable right now: {exc.__class__.__name__}. The verified sales metrics remain available below."
