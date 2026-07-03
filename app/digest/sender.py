import logging

import requests

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.digest.builder import build_digest_text

logger = logging.getLogger(__name__)
settings = get_settings()


def send_digest() -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.info("Telegram is not configured; digest skipped")
        return

    with SessionLocal() as db:
        text = build_digest_text(db)

    response = requests.post(
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
        json={"chat_id": settings.telegram_chat_id, "text": text},
        timeout=30,
    )
    response.raise_for_status()
