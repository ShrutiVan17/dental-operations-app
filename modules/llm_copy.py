# modules/llm_copy.py
"""
Generate subject + body for appointment emails using Ollama (if available),
with a clean, realistic fallback template if Ollama is not running.
"""

from __future__ import annotations
import os
import requests


# ---- Ollama config ----
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

CLINIC_NAME = "HappyClinic Dental"   # change this if you want


SYSTEM = (
    "You are an assistant for a modern dental clinic called "
    f"'{CLINIC_NAME}'. "
    "Write short, friendly, professional plain-text emails for patients. "
    "Always sound human and natural, no placeholders like [Your Name] "
    "or [Clinic Name]. Use the patient name and the appointment details "
    "given in the context. "
    "Address the patient by their name (e.g., 'Hi Shruzz Patel,'). "
    "Do NOT thank them for booking with 'our dental clinic'; instead, "
    "speak as the actual clinic. "
    "No emojis, no markdown, just plain text. "
    "Return the answer in this exact format:\n"
    "SUBJECT: <one short subject line>\n"
    "BODY:\n<short multi-line body>\n"
)


def _call_ollama(prompt: str, timeout: int = 40) -> str:
    """Call Ollama /api/generate with the combined system + user prompt."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM}\n\n{prompt}",
            "stream": False,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def draft_email_copy(context: dict) -> tuple[str, str]:
    """
    context = {
      'type': 'Hygiene' | 'Restorative',
      'patient': 'Full Name',
      'date': 'YYYY-MM-DD',
      'time': 'HH:MM',
      'status': 'scheduled' | 'rescheduled' | 'cancelled'
    }
    returns (subject, body)
    """

    t = context.get("type", "Dental")
    p = context.get("patient", "Patient")
    d = context.get("date")
    tm = context.get("time")
    status = context.get("status", "scheduled")

    # ---- Build a clear user prompt for the LLM ----
    user_prompt = (
        "Write a confirmation email for a dental appointment.\n"
        f"Clinic name: {CLINIC_NAME}\n"
        f"Patient name: {p}\n"
        f"Appointment type: {t}\n"
        f"Status: {status}\n"
        f"Date: {d}\n"
        f"Time: {tm}\n\n"
        "Rules:\n"
        "- Subject should mention appointment type and date/time briefly.\n"
        "- Body should be 3–6 short lines.\n"
        "- Confirm the appointment details clearly.\n"
        "- If status is 'rescheduled', explain the new time.\n"
        "- If status is 'cancelled', confirm the cancellation.\n"
        "- No placeholders like [Your Name] or [Clinic Name]. "
        "Sign off as the clinic team (e.g., 'HappyClinic Dental Team')."
    )

    try:
        raw = _call_ollama(user_prompt)
        subj = "Appointment update"
        body = raw

        if "SUBJECT:" in raw and "BODY:" in raw:
            subj = raw.split("BODY:", 1)[0].replace("SUBJECT:", "").strip()
            body = raw.split("BODY:", 1)[1].strip()

        # Very short sanity check – if subject/body look empty, fall back
        if not subj or not body:
            raise ValueError("Empty subject/body from Ollama")

        return subj, body

    except Exception:
        # -------- Fallback template (if Ollama is not running) --------
        d_str = d or "the scheduled date"
        tm_str = tm or "the scheduled time"
        t_lower = t.lower()

        if status == "cancelled":
            subject = f"{t} appointment on {d_str} at {tm_str} cancelled"
            body = f"""Hi {p},

Your {t_lower} appointment at {CLINIC_NAME} for {tm_str} on {d_str} has been cancelled as requested.

If this was a mistake, reply to this email and we’ll be happy to book a new appointment.

Best regards,
{CLINIC_NAME} Team
"""
        elif status == "rescheduled":
            subject = f"{t} appointment rescheduled to {tm_str} on {d_str}"
            body = f"""Hi {p},

We’ve updated your {t_lower} appointment at {CLINIC_NAME}. Your new time is {tm_str} on {d_str}.

If this change doesn’t work for you, reply to this email and we can find another slot.

Best regards,
{CLINIC_NAME} Team
"""
        else:  # scheduled
            subject = f"{t} appointment confirmed for {tm_str} on {d_str}"
            body = f"""Hi {p},

This is a confirmation that your {t_lower} appointment at {CLINIC_NAME} is booked for {tm_str} on {d_str}.

If you need to reschedule or cancel, please reply to this email and we will be happy to help.

Best regards,
{CLINIC_NAME} Team
"""

        return subject, body
