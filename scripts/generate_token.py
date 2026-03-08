"""
scripts/generate_token.py — Automated Fyers access token generation.

Uses TOTP key + PIN to fully automate the OAuth flow.
Saves the token to .env (FYERS_ACCESS_TOKEN) and a cache file.
Token is valid for ~24 hours.
"""

from __future__ import annotations

import sys
import os
import json
import base64
import time
from datetime import date, datetime

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pyotp
import requests
from urllib.parse import urlparse, parse_qs
from fyers_apiv3 import fyersModel
from config import settings


TOKEN_CACHE = os.path.join(_project_root, "storage", "broker_token.json")


def _encode(value: str) -> str:
    """Base64-encode a string (Fyers expects this for credentials)."""
    return base64.b64encode(value.encode("ascii")).decode("ascii")


def _load_cached_token() -> str | None:
    """Return today's cached token if it exists."""
    try:
        with open(TOKEN_CACHE, "r") as f:
            data = json.load(f)
        today = date.today().strftime("%Y-%m-%d")
        if today in data:
            print(f"[auth] Found cached token for {today}")
            return data[today]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _save_token(token: str) -> None:
    """Cache today's token to disk."""
    today = date.today().strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(TOKEN_CACHE), exist_ok=True)
    with open(TOKEN_CACHE, "w") as f:
        json.dump({today: token}, f)
    print(f"[auth] Token cached to {TOKEN_CACHE}")


def _update_env_file(token: str) -> None:
    """Update FYERS_ACCESS_TOKEN in .env file."""
    env_path = os.path.join(_project_root, ".env")
    if not os.path.exists(env_path):
        print(f"[auth] WARNING: .env not found at {env_path}")
        return

    with open(env_path, "r") as f:
        lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith("FYERS_ACCESS_TOKEN"):
            lines[i] = f"FYERS_ACCESS_TOKEN={token}\n"
            found = True
            break

    if not found:
        lines.append(f"FYERS_ACCESS_TOKEN={token}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)
    print(f"[auth] .env updated with new access token")


def generate_token() -> str:
    """Full automated Fyers login flow. Returns access_token."""

    # Check cache first
    cached = _load_cached_token()
    if cached:
        return cached

    # Validate required credentials
    if not settings.FYERS_USER_ID:
        raise ValueError("FYERS_USER_ID not set in .env")
    if not settings.FYERS_TOTP_KEY:
        raise ValueError("FYERS_TOTP_KEY not set in .env")
    if not settings.FYERS_PIN:
        raise ValueError("FYERS_PIN not set in .env")
    if not settings.FYERS_APP_ID:
        raise ValueError("FYERS_APP_ID not set in .env")
    if not settings.FYERS_SECRET_KEY:
        raise ValueError("FYERS_SECRET_KEY not set in .env")

    print(f"[auth] Generating fresh token for {settings.FYERS_USER_ID}...")

    # Step 1: Send login OTP
    print("[auth] Step 1/4: Sending login OTP...")
    url = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
    resp = requests.post(url, json={
        "fy_id": _encode(settings.FYERS_USER_ID),
        "app_id": "2",
    })
    resp.raise_for_status()
    res1 = resp.json()

    if res1.get("code") != 200 and res1.get("s") != "ok":
        raise RuntimeError(f"Step 1 failed: {res1}")

    request_key = res1["request_key"]

    # Step 2: Verify TOTP
    # Wait if we're near the end of a TOTP window to avoid expiry
    if datetime.now().second % 30 > 26:
        print("[auth] Waiting for fresh TOTP window...")
        time.sleep(5)

    print("[auth] Step 2/4: Verifying TOTP...")
    totp = pyotp.TOTP(settings.FYERS_TOTP_KEY)
    otp_code = totp.now()

    url = "https://api-t2.fyers.in/vagator/v2/verify_otp"
    resp = requests.post(url, json={
        "request_key": request_key,
        "otp": otp_code,
    })
    resp.raise_for_status()
    res2 = resp.json()

    if "request_key" not in res2:
        raise RuntimeError(f"Step 2 failed: {res2}")

    # Step 3: Verify PIN
    print("[auth] Step 3/4: Verifying PIN...")
    session = requests.Session()
    url = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
    resp = session.post(url, json={
        "request_key": res2["request_key"],
        "identity_type": "pin",
        "identifier": _encode(settings.FYERS_PIN),
    })
    resp.raise_for_status()
    res3 = resp.json()

    if "data" not in res3 or "access_token" not in res3.get("data", {}):
        raise RuntimeError(f"Step 3 failed: {res3}")

    # Set bearer token for the session
    session.headers.update({
        "authorization": f"Bearer {res3['data']['access_token']}"
    })

       # Step 4: Get auth code via token endpoint
    print("[auth] Step 4/4: Exchanging auth code for access token...")
    app_id_short = settings.FYERS_APP_ID.split("-")[0] if "-" in settings.FYERS_APP_ID else settings.FYERS_APP_ID

    url = "https://api-t1.fyers.in/api/v3/token"
    payload = {
        "fyers_id": settings.FYERS_USER_ID,
        "app_id": app_id_short,
        "redirect_uri": settings.FYERS_REDIRECT_URI,
        "appType": "100",
        "code_challenge": "",
        "state": "None",
        "scope": "",
        "nonce": "",
        "response_type": "code",
        "create_cookie": True,
    }
    resp = session.post(url, json=payload)
    resp.raise_for_status()
    res4 = resp.json()

    # Extract auth_code — try multiple response formats
    auth_code = None

    # Format 1: Url field with auth_code in query params
    if "Url" in res4:
        parsed = urlparse(res4["Url"])
        qs = parse_qs(parsed.query)
        if "auth_code" in qs:
            auth_code = qs["auth_code"][0]

    # Format 2: data.url field
    if not auth_code and "data" in res4:
        data = res4["data"]
        # Check for URL in data
        for key in ("Url", "url", "redirectUrl"):
            if key in data and data[key] and "auth_code" in str(data[key]):
                parsed = urlparse(data[key])
                qs = parse_qs(parsed.query)
                if "auth_code" in qs:
                    auth_code = qs["auth_code"][0]
                    break

    # Format 3: Use the auth token to make the redirect call manually
    if not auth_code and res4.get("s") == "ok" and "data" in res4:
        # The response contains redirect info — we need to follow it
        data = res4["data"]
        redirect_url = data.get("redirectUrl", settings.FYERS_REDIRECT_URI)

        # Make the authorize call with the bearer token from step 3
        url2 = "https://api-t1.fyers.in/api/v3/auth-code"
        resp2 = session.post(url2, json={
            "fyers_id": settings.FYERS_USER_ID,
            "app_id": app_id_short,
            "redirect_uri": redirect_url,
            "appType": "100",
            "code_challenge": "",
            "state": "None",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        })

        if resp2.status_code == 200:
            res5 = resp2.json()
            if "Url" in res5:
                parsed = urlparse(res5["Url"])
                qs = parse_qs(parsed.query)
                if "auth_code" in qs:
                    auth_code = qs["auth_code"][0]
            elif "data" in res5 and "Url" in res5.get("data", {}):
                parsed = urlparse(res5["data"]["Url"])
                qs = parse_qs(parsed.query)
                if "auth_code" in qs:
                    auth_code = qs["auth_code"][0]

    # Format 4: Construct the session model and try generate_authcode
    if not auth_code:
        try:
            token_session = fyersModel.SessionModel(
                client_id=settings.FYERS_APP_ID,
                secret_key=settings.FYERS_SECRET_KEY,
                redirect_uri=settings.FYERS_REDIRECT_URI,
                response_type="code",
                grant_type="authorization_code",
            )
            auth_url = token_session.generate_authcode()
            print(f"[auth] Auth URL generated: {auth_url[:80]}...")

            # Try to follow the auth URL with our authenticated session
            resp_auth = session.get(auth_url, allow_redirects=False)
            if resp_auth.status_code in (301, 302, 303, 307):
                location = resp_auth.headers.get("Location", "")
                parsed = urlparse(location)
                qs = parse_qs(parsed.query)
                if "auth_code" in qs:
                    auth_code = qs["auth_code"][0]
        except Exception as e:
            print(f"[auth] Format 4 attempt failed: {e}")

    # Format 5: The 'auth' field in data IS the auth_code (JWT format)
    if not auth_code and res4.get("s") == "ok" and "data" in res4:
        auth_jwt = res4["data"].get("auth")
        if auth_jwt:
            print(f"[auth] Trying 'auth' JWT as auth_code...")
            auth_code = auth_jwt

    if not auth_code:
        raise RuntimeError(
            f"Could not extract auth_code from any response format.\n"
            f"Last response: {json.dumps(res4, indent=2)[:500]}"
        )

    # Exchange auth code for final access token
    token_session = fyersModel.SessionModel(
        client_id=settings.FYERS_APP_ID,
        secret_key=settings.FYERS_SECRET_KEY,
        redirect_uri=settings.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    token_session.set_token(auth_code)
    response = token_session.generate_token()

    if "access_token" not in response:
        raise RuntimeError(f"Token exchange failed: {response}")

    access_token = response["access_token"]

    # Cache and update .env
    _save_token(access_token)
    _update_env_file(access_token)

    print(f"[auth] Token generated successfully!")
    return access_token


def verify_token(token: str) -> bool:
    """Quick check: does the token work?"""
    fyers = fyersModel.FyersModel(
        client_id=settings.FYERS_APP_ID,
        is_async=False,
        token=token,
        log_path="",
    )
    profile = fyers.get_profile()
    if profile.get("s") == "ok":
        name = profile.get("data", {}).get("name", "Unknown")
        print(f"[auth] Token verified — logged in as: {name}")
        return True
    else:
        print(f"[auth] Token verification failed: {profile}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  Fyers Access Token Generator")
    print("=" * 50)
    print()

    try:
        token = generate_token()
        print()
        print(f"Access Token: {token[:20]}...{token[-10:]}")
        print()
        verify_token(token)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nCheck your .env credentials:")
        print("  FYERS_USER_ID, FYERS_APP_ID, FYERS_SECRET_KEY")
        print("  FYERS_TOTP_KEY, FYERS_PIN")
        sys.exit(1)