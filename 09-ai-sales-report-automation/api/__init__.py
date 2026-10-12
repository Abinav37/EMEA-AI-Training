"""API integrations for the sales report automation project."""

from .openrouter_client import build_report_prompt, generate_ai_report

__all__ = ["build_report_prompt", "generate_ai_report"]
