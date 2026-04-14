from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass
class DeliveryResult:
    ok: bool
    provider: str
    destination: str
    detail: str
    external_id: str | None = None


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def send_resend_email(to_email: str, subject: str, text: str, html: str | None = None) -> DeliveryResult:
    api_key = _read_env("RESEND_API_KEY")
    from_email = _read_env("RESEND_FROM_EMAIL")
    if not api_key or not from_email:
        return DeliveryResult(False, "resend", to_email, "Missing RESEND_API_KEY or RESEND_FROM_EMAIL")

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": text,
    }
    if html:
        payload["html"] = html

    req = Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return DeliveryResult(True, "resend", to_email, "Email sent", external_id=body.get("id"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return DeliveryResult(False, "resend", to_email, f"HTTP {exc.code}: {detail}")
    except URLError as exc:
        return DeliveryResult(False, "resend", to_email, f"Network error: {exc}")


def send_twilio_sms(to_phone: str, text: str) -> DeliveryResult:
    account_sid = _read_env("TWILIO_ACCOUNT_SID")
    auth_token = _read_env("TWILIO_AUTH_TOKEN")
    from_phone = _read_env("TWILIO_FROM_PHONE")
    if not account_sid or not auth_token or not from_phone:
        return DeliveryResult(False, "twilio", to_phone, "Missing TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, or TWILIO_FROM_PHONE")

    body = urlencode(
        {
            "From": from_phone,
            "To": to_phone,
            "Body": text,
        }
    ).encode("utf-8")
    token = base64.b64encode(f"{account_sid}:{auth_token}".encode("utf-8")).decode("ascii")
    req = Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
        data=body,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return DeliveryResult(True, "twilio", to_phone, "SMS sent", external_id=payload.get("sid"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return DeliveryResult(False, "twilio", to_phone, f"HTTP {exc.code}: {detail}")
    except URLError as exc:
        return DeliveryResult(False, "twilio", to_phone, f"Network error: {exc}")


def send_notification(*, channel: str, destination: str, subject: str, text: str, html: Optional[str] = None) -> DeliveryResult:
    if channel == "email":
        return send_resend_email(destination, subject=subject, text=text, html=html)
    if channel == "sms":
        return send_twilio_sms(destination, text=text)
    return DeliveryResult(False, "unknown", destination, f"Unsupported channel: {channel}")
