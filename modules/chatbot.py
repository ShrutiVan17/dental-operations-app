# modules/chatbot.py — data-aware Q&A with baseline = latest date in CSV

from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd

from .llm_helper import _ollama_generate, OLLAMA_MODEL


def _get_date_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return a normalized daily date Series, if possible."""
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
    elif "DateKpi" in df.columns:
        d = pd.to_datetime(df["DateKpi"], errors="coerce")
    else:
        return None
    d = d.dt.floor("D")
    if d.isna().all():
        return None
    return d


def _get_baseline_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Baseline = latest date in the dataset (used instead of 'today')."""
    d = _get_date_series(df)
    if d is None:
        return None
    return d.max()


def build_context_cards(df: pd.DataFrame) -> Dict[str, str]:
    """Compute compact stats we can safely pass to the model as context."""
    ctx: Dict[str, str] = {}

    d = _get_date_series(df)
    if d is not None:
        ctx["date_range"] = f"{d.min().date()} → {d.max().date()}"
        ctx["baseline_date"] = str(d.max().date())

    # simple numeric summaries
    for col in ["Collections", "CollectionsPatient", "Profit"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            ctx[f"{col}_total"] = float(s.sum())
            ctx[f"{col}_avg_per_row"] = float(s.mean())

    for col in ["VisitsCompleted", "VisitsNoShow"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            ctx[f"{col}_total"] = float(s.sum())

    return ctx


def _resolve_metric(df: pd.DataFrame, question: str) -> Optional[str]:
    """Pick which numeric column to treat as 'revenue'."""
    q = question.lower()

    # Prefer Collections as revenue metric
    if "Collections" in df.columns:
        return "Collections"
    if "Profit" in df.columns:
        return "Profit"

    # If user mentions an exact column name, use it
    for col in df.columns:
        if col.lower() in q and pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None


def _sum_window(df: pd.DataFrame, metric: str, start, end) -> float:
    d = _get_date_series(df)
    if d is None:
        return float("nan")
    s = pd.to_numeric(df[metric], errors="coerce")
    mask = (d >= start) & (d <= end)
    return float(s[mask].sum())


SYSTEM = (
    "You are a data assistant for a dental clinic. "
    "Answer using ONLY the numeric context provided. "
    "When dates are involved, interpret phrases like 'last month' or 'last week' "
    "relative to the `baseline_date` from the data, not today's real-world date. "
    "Be concise; explain in simple, non-technical language."
)


def chat_answer(question: str, df: pd.DataFrame, ctx: Dict[str, str]) -> str:
    """Answer simple time-window questions directly, else fall back to LLM."""

    q = question.strip()
    q_lower = q.lower()

    baseline = _get_baseline_date(df)
    metric = _resolve_metric(df, q)

    # Try to directly answer 'last month / last week' style questions
    if baseline is not None and metric is not None:
        window_days = None
        label = None

        if "last month" in q_lower or "previous month" in q_lower:
            window_days, label = 30, "last month"
        elif "last 30 days" in q_lower:
            window_days, label = 30, "last 30 days"
        elif "last week" in q_lower:
            window_days, label = 7, "last week"
        elif "last 7 days" in q_lower:
            window_days, label = 7, "last 7 days"
        elif "last quarter" in q_lower or "last 3 months" in q_lower:
            window_days, label = 90, "last 3 months"

        if window_days is not None:
            start = (baseline - pd.Timedelta(days=window_days)).normalize()
            end = baseline
            total = _sum_window(df, metric, start, end)

            if not pd.isna(total):
                return (
                    f"For {label}, based on your data from {start.date()} to {end.date()} "
                    f"(ending at the latest date {baseline.date()}), "
                    f"total {metric} was about **${total:,.0f}**. "
                    f"This is calculated directly from your file, not guessed."
                )

    # Fallback: general LLM answer using compact context
    enriched_ctx = dict(ctx)
    if baseline is not None:
        enriched_ctx["baseline_date"] = str(baseline.date())
    cards_json = json.dumps(enriched_ctx)[:3000]

    prompt = (
        f"{SYSTEM}\n\n"
        f"CONTEXT (JSON): {cards_json}\n\n"
        f"QUESTION: {question}\n\n"
        "If the context lacks specificity, ask the user for a date range or which metric they care about "
        "(for example, 'Collections' or 'Profit'). "
        "Remember that 'last month' and similar phrases are relative to `baseline_date`, not today."
    )
    return _ollama_generate(prompt, model=OLLAMA_MODEL)
