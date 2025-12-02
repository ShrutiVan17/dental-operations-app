# modules/mailer.py
from __future__ import annotations

import requests
import smtplib
import ssl
from email.message import EmailMessage

# ---------------------------------------------------
# SIMPLE CLASS PROJECT SETUP:
#  - Hard-code your Resend API key here
#  - Hard-code the "from" email
# ---------------------------------------------------

RESEND_API_KEY = "re_TiKBWdsv_BU7xfj5ciy4BFyfwVqYKzUkK"     
FROM_EMAIL     = "onboarding@resend.dev"         # Resend's free sender

# (optional) SMTP fallback – leave as None if you don't use it
SMTP_HOST = None
SMTP_PORT = 587
SMTP_USER = None
SMTP_PASS = None


def send_email(to_email: str, subject: str, body_text: str) -> tuple[bool, str]:
    """
    Send an email using Resend if RESEND_API_KEY is set.
    Falls back to SMTP if SMTP_* are set.
    Returns: (success_flag, message)
    """

    # 1) Try Resend first
    if RESEND_API_KEY:
        try:
            r = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": FROM_EMAIL,
                    "to": [to_email],
                    "subject": subject,
                    "text": body_text,
                },
                timeout=20,
            )
            r.raise_for_status()
            return True, "Sent via Resend"
        except Exception as e:
            return False, f"Resend error: {e}"

    # 2) Optional SMTP fallback (you probably won't use this)
    if SMTP_HOST and SMTP_USER and SMTP_PASS:
        try:
            msg = EmailMessage()
            msg["From"] = FROM_EMAIL
            msg["To"] = to_email
            msg["Subject"] = subject
            msg.set_content(body_text)

            ctx = ssl.create_default_context()
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls(context=ctx)
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)

            return True, "Sent via SMTP"
        except Exception as e:
            return False, f"SMTP error: {e}"

    # 3) Nothing configured
    return False, "No email provider configured (RESEND_API_KEY is empty)"
