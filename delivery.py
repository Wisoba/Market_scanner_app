from __future__ import annotations

import base64
import time
import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
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


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _read_apns_private_key() -> str | None:
    raw_key = _read_env("APNS_AUTH_KEY")
    if raw_key:
        return raw_key.replace("\\n", "\n")

    key_path = _read_env("APNS_AUTH_KEY_PATH")
    if not key_path:
        return None
    try:
        with open(key_path, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _apns_jwt() -> tuple[str | None, str | None]:
    key_id = _read_env("APNS_KEY_ID")
    team_id = _read_env("APNS_TEAM_ID")
    private_key = _read_apns_private_key()
    if not key_id or not team_id or not private_key:
        return None, "Missing APNS_KEY_ID, APNS_TEAM_ID, or APNS_AUTH_KEY/APNS_AUTH_KEY_PATH"

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except Exception:
        return None, "Missing Python package: cryptography"

    try:
        key = serialization.load_pem_private_key(private_key.encode("utf-8"), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            return None, "APNs auth key is not an EC private key"

        header = {"alg": "ES256", "kid": key_id}
        claims = {"iss": team_id, "iat": int(time.time())}
        signing_input = f"{_base64url(json.dumps(header, separators=(',', ':')).encode())}.{_base64url(json.dumps(claims, separators=(',', ':')).encode())}"
        signature = key.sign(signing_input.encode("ascii"), ec.ECDSA(hashes.SHA256()))

        from cryptography.hazmat.primitives.asymmetric import utils

        r, s = utils.decode_dss_signature(signature)
        raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return f"{signing_input}.{_base64url(raw_signature)}", None
    except Exception as exc:
        return None, f"Could not create APNs JWT: {exc}"


def send_apns_push(device_token: str, subject: str, text: str) -> DeliveryResult:
    topic = _read_env("APNS_TOPIC") or _read_env("APNS_BUNDLE_ID")
    if not topic:
        return DeliveryResult(False, "apns", device_token, "Missing APNS_TOPIC/APNS_BUNDLE_ID")

    token, token_error = _apns_jwt()
    if token_error:
        return DeliveryResult(False, "apns", device_token, token_error)

    environment = (_read_env("APNS_ENVIRONMENT") or "production").lower()
    host = "api.sandbox.push.apple.com" if environment == "sandbox" else "api.push.apple.com"
    payload = {
        "aps": {
            "alert": {
                "title": subject,
                "body": text,
            },
            "sound": "default",
        },
        "source": "gace_scan",
    }
    req = Request(
        f"https://{host}/3/device/{device_token}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "authorization": f"bearer {token}",
            "apns-topic": topic,
            "apns-push-type": "alert",
            "apns-priority": "10",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            apns_id = resp.headers.get("apns-id")
        return DeliveryResult(True, "apns", device_token, "Push sent", external_id=apns_id)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return DeliveryResult(False, "apns", device_token, f"HTTP {exc.code}: {detail}")
    except URLError as exc:
        return DeliveryResult(False, "apns", device_token, f"Network error: {exc}")


def send_notification(*, channel: str, destination: str, subject: str, text: str, html: Optional[str] = None) -> DeliveryResult:
    if channel == "push":
        return send_apns_push(destination, subject=subject, text=text)
    if channel == "email":
        return send_resend_email(destination, subject=subject, text=text, html=html)
    return DeliveryResult(False, "unknown", destination, f"Unsupported channel: {channel}")
