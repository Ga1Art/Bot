from __future__ import annotations

from pathlib import Path

import requests
from sqlalchemy import or_
from sqlalchemy import text

from app.core.config import get_settings
from app.db.models import Lead
from app.db.session import SessionLocal
from app.normalizers.region import TARGET_EUROPEAN_RUSSIA_REGIONS


def _is_filled(value: str, placeholder: str = "replace_me") -> bool:
    return bool(value and value.strip() and value.strip() != placeholder)


def _check_api(settings) -> tuple[str, str]:
    url = f"http://127.0.0.1:{settings.app_port}/openapi.json"
    try:
        response = requests.get(url, timeout=2)
    except requests.RequestException:
        return "WARN", f"API is not running on {url}"

    if response.status_code != 200:
        return "WARN", f"API responded with HTTP {response.status_code} on {url}"

    paths = response.json().get("paths", {})
    if "/leads/hot" not in paths:
        return "WARN", f"API is running on port {settings.app_port}, but it looks like an old process"

    return "OK", f"API is running on port {settings.app_port}"


def main() -> None:
    settings = get_settings()
    checks: list[tuple[str, str]] = []

    checks.append(("OK", f"Environment: {settings.app_env}"))
    checks.append(("OK", f"API bind: {settings.app_host}:{settings.app_port}"))
    checks.append(("OK", f"Priority regions: {', '.join(settings.priority_regions)}"))

    if _is_filled(settings.telegram_bot_token) and _is_filled(settings.telegram_chat_id):
        checks.append(("OK", "Telegram bot token and chat id are configured"))
    else:
        checks.append(("WARN", "Telegram is not configured: fill TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"))

    if settings.google_sheets_credentials_file and settings.google_sheets_spreadsheet_id:
        credentials_path = Path(settings.google_sheets_credentials_file)
        if credentials_path.exists():
            checks.append(("OK", "Google Sheets credentials file exists"))
        else:
            checks.append(("FAIL", f"Google Sheets credentials file not found: {credentials_path}"))
    else:
        checks.append(("WARN", "Google Sheets is not configured"))

    try:
        with SessionLocal() as db:
            db.execute(text("select 1"))
            revision = db.execute(text("select version_num from alembic_version")).scalar_one_or_none()
            total = db.query(Lead).count()
            hot = (
                db.query(Lead)
                .filter(Lead.status.in_(("new", "in_work")))
                .filter(Lead.priority.in_(("A", "B")))
                .filter(
                    or_(
                        Lead.region.in_(settings.priority_regions),
                        *[Lead.region.ilike(f"%{region}%") for region in TARGET_EUROPEAN_RUSSIA_REGIONS],
                    )
                )
                .count()
            )
        checks.append(("OK", "Database connection works"))
        checks.append(("OK", f"Alembic revision: {revision or 'unknown'}"))
        checks.append(("OK", f"Leads in database: {total}, hot prospects: {hot}"))
    except Exception as exc:
        checks.append(("FAIL", f"Database check failed: {type(exc).__name__}: {exc}"))

    checks.append(_check_api(settings))

    for status, message in checks:
        print(f"[{status}] {message}")

    if any(status == "FAIL" for status, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
