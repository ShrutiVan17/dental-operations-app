# modules/llm_helper.py  — Ollama only (no OpenAI needed)

from __future__ import annotations
import os, time, json, requests
from pathlib import Path
from typing import Optional

# =========================
# Load .env if present
# =========================
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore

    # Load from any .env on the path
    load_dotenv(find_dotenv(), override=True)
    # And from project root (one level above /modules)
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
except Exception:
    # If python-dotenv is not installed, just skip
    pass

# =========================
# Ollama Config
# =========================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

SYSTEM_PROMPT = (
    "You are a marketing strategist for dental clinics. "
    "Given KPI context and (optional) scraped excerpts, produce:\n"
    "1) 5–8 prioritized strategies (title, why it fits, step-by-step actions, expected impact, cost level).\n"
    "2) 2 sample campaign messages (one SMS, one email) with strong CTAs.\n"
    "Be actionable, compliant, and measurable. Avoid copyrighted book content."
)


# =========================
# Low-level Ollama caller
# =========================
def _ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: int = 600,
    max_tokens: int = 400,
    retries: int = 2,
) -> str:
    """
    Call local Ollama with streaming + retry and a capped output length
    (helps avoid long waits/timeouts on first runs).
    """
    m = model or OLLAMA_MODEL
    u = (url or OLLAMA_URL).rstrip("/") + "/api/generate"
    last_err: Optional[Exception] = None

    for _ in range(retries + 1):
        try:
            with requests.post(
                u,
                json={
                    "model": m,
                    "prompt": prompt,
                    "stream": True,               # stream tokens for quicker first output
                    "options": {
                        "num_predict": max_tokens,  # cap output to return faster
                        "temperature": 0.4,
                        "repeat_penalty": 1.1,
                    },
                },
                timeout=timeout,
                stream=True,
            ) as r:
                r.raise_for_status()
                chunks: list[str] = []
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("done"):
                            break
                        piece = obj.get("response")
                        if piece:
                            chunks.append(piece)
                    except Exception:
                        # ignore keep-alives or partial lines
                        continue
                text = "".join(chunks).strip()
                return text or "(No response from local model.)"

        except Exception as e:
            last_err = e
            time.sleep(2)  # brief backoff, then retry

    return f"(Local LLM error: {last_err})"


# =========================
# High-level helper used by app.py
# =========================
def draft_strategies(kpi_context: dict, scraped_text: str) -> str:
    """
    Public entry used by the app.

    kpi_context comes from app.py and may include:
      - no_show_risk_avg
      - predicted_collections_shortfall
      - forecast_open_slots_gap
      - goal
      - style
      - time_horizon
      - avg_collections_per_slot
      - estimated_monthly_upside
      - insights (list of strings)

    scraped_text is a long string of text scraped from public URLs
    (can be empty).
    """

    # ----- unpack core fields -----
    goal = kpi_context.get("goal", "Cut no-shows")
    style = kpi_context.get("style", "Detailed playbook")
    horizon = kpi_context.get("time_horizon", "Next 1 month")
    insights = kpi_context.get("insights", [])

    # Small, clean JSON summary of the numeric KPIs
    numeric_summary = {
        "no_show_rate": kpi_context.get("no_show_risk_avg"),
        "daily_collections_shortfall": kpi_context.get("predicted_collections_shortfall"),
        "open_slots_per_day": kpi_context.get("forecast_open_slots_gap"),
        "avg_collections_per_slot": kpi_context.get("avg_collections_per_slot"),
        "estimated_monthly_upside": kpi_context.get("estimated_monthly_upside"),
    }

    insight_text = "\n".join(f"- {t}" for t in insights) if insights else "- (no extra insights)"

    # =========================
    # Style-specific instructions
    # =========================
    if style == "Quick checklist":
        style_block = (
            "WRITING INSTRUCTIONS:\n"
            "- Write a SHORT, PUNCHY CHECKLIST.\n"
            "- Use 3–5 headings (e.g. 'Reminders', 'Front-desk scripting', 'Scheduling rules').\n"
            "- Under each heading, give 3–6 bullet points.\n"
            "- Each bullet should be 1–2 lines and very concrete.\n"
            "- No long paragraphs, no story, no conclusion essay.\n"
        )
    elif style == "Email to my team":
        style_block = (
            "WRITING INSTRUCTIONS:\n"
            "- Write the answer as a FRIENDLY but PROFESSIONAL EMAIL from the owner to the team.\n"
            "- Start with 'Hi team,'.\n"
            "- In the first short paragraph, explain the situation and the goal in plain language.\n"
            "- Then include a bullet list of concrete actions the team should take starting this week.\n"
            "- Close with a short motivating line and sign off as 'Doctor'.\n"
            "- Do NOT include a subject line.\n"
        )
    else:  # Detailed playbook (default)
        style_block = (
            "WRITING INSTRUCTIONS:\n"
            "- Write a DETAILED PLAYBOOK with clear section headings.\n"
            "- Use numbered sections (e.g. '1. Fix the reminder system').\n"
            "- Each section should have 1–3 short paragraphs plus 3–5 bullet points.\n"
            "- Use simple sentences and concrete actions.\n"
            "- End with a 'Next 7 days action list' summarising the top 5 actions.\n"
        )

    # =========================
    # Goal-specific instructions
    # =========================
    g_lower = str(goal).lower()
    if "no-show" in g_lower or "no shows" in g_lower:
        goal_block = (
            "FOCUS AREA:\n"
            "- Prioritise reducing no-shows and improving confirmation workflows.\n"
            "- Include ideas for reminders, confirmations, backup patients and overbooking rules.\n"
        )
    elif "hygiene" in g_lower:
        goal_block = (
            "FOCUS AREA:\n"
            "- Prioritise filling the hygiene schedule and reactivating overdue patients.\n"
            "- Include ideas for recall systems, overdue lists and short-notice list management.\n"
        )
    elif "high-value" in g_lower or "crowns" in g_lower or "implants" in g_lower:
        goal_block = (
            "FOCUS AREA:\n"
            "- Prioritise increasing acceptance and scheduling of high-value treatments "
            "(crowns, implants, large restorative cases) without feeling pushy.\n"
            "- Include ideas for case presentation, financing, follow-ups and tracking.\n"
        )
    elif "recall" in g_lower or "reactivation" in g_lower:
        goal_block = (
            "FOCUS AREA:\n"
            "- Prioritise recall and reactivation: getting overdue patients back, improving recall systems, "
            "and building a consistent follow-up rhythm.\n"
        )
    else:
        goal_block = (
            "FOCUS AREA:\n"
            "- Provide a balanced mix of reducing no-shows, improving schedule fill and lifting production.\n"
        )

    # =========================
    # External scraped content (optional)
    # =========================
    # Trim scraped content so prompt doesn't explode
    notes = (scraped_text or "")[:1500]
    if notes:
        external_block = (
            "SCRAPED PUBLIC CONTENT (ideas source – do NOT summarise, adapt only):\n"
            + notes
            + "\n---- END OF SCRAPED CONTENT ----\n"
        )
    else:
        external_block = "SCRAPED PUBLIC CONTENT:\n- (none provided)\n"

    # =========================
    # Final user prompt
    # =========================
    user_prompt = (
        f"PRACTICE KPI SNAPSHOT (JSON):\n{json.dumps(numeric_summary, indent=2)}\n\n"
        f"MAIN GOAL FOR THIS PLAN:\n- {goal}\n\n"
        f"TIME HORIZON:\n- {horizon}\n\n"
        f"HUMAN INSIGHTS ABOUT THE SITUATION:\n{insight_text}\n\n"
        f"{goal_block}\n"
        f"{style_block}\n"
        f"{external_block}\n"
        "Now write the full strategy as requested, following ALL writing instructions."
    )

    # Combine with system prompt and call Ollama
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    return _ollama_generate(full_prompt)
