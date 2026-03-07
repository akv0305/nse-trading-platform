"""
Fyers OAuth authentication helper.
Generates access token for the trading session.

Usage:
  python scripts/fyers_auth.py
"""

from __future__ import annotations

import logging
import webbrowser

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run Fyers OAuth flow to generate access token."""
    from fyers_apiv3 import fyersModel

    if not settings.FYERS_APP_ID or not settings.FYERS_SECRET_KEY:
        logger.error("Set FYERS_APP_ID and FYERS_SECRET_KEY in .env first")
        return

    session = fyersModel.SessionModel(
        client_id=settings.FYERS_APP_ID,
        secret_key=settings.FYERS_SECRET_KEY,
        redirect_uri=settings.FYERS_REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )

    auth_url = session.generate_authcode()
    logger.info(f"Opening browser for Fyers login...")
    logger.info(f"Auth URL: {auth_url}")
    webbrowser.open(auth_url)

    auth_code = input("\nPaste the auth_code from the redirect URL: ").strip()

    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") == "ok":
        token = response["access_token"]
        logger.info(f"\nAccess Token (add to .env as FYERS_ACCESS_TOKEN):\n{token}")

        # Optionally save to file
        with open("access_token.txt", "w") as f:
            f.write(token)
        logger.info("Token also saved to access_token.txt")
    else:
        logger.error(f"Token generation failed: {response}")


if __name__ == "__main__":
    main()
