import logging
from datetime import datetime

import requests

from app.bot.keyboards import lead_actions_keyboard
from app.bot.templates import render_lead_card
from app.db.models import Lead
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TelegramService:
    def is_configured(self) -> bool:
        return bool(settings.telegram_bot_token and settings.telegram_chat_id)

    def notify_new_priority_leads(self, leads: list[Lead]) -> None:
        if not self.is_configured():
            logger.info("Telegram is not configured; skipping lead notifications")
            return

        priority_leads = [
            lead for lead in leads if lead.priority in {"A", "B"} and lead.notified_at is None
        ]
        if not priority_leads:
            return

        for lead in priority_leads:
            text = render_lead_card(
                title=lead.title,
                region=lead.region,
                url=lead.url,
                priority=lead.priority,
                source_name=lead.source_name,
                customer_name=lead.customer_name,
                budget_max=lead.budget_max,
                deadline_at=lead.deadline_at.strftime("%d.%m.%Y %H:%M") if lead.deadline_at else None,
                score=lead.relevance_score,
            )
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                    json={
                        "chat_id": settings.telegram_chat_id,
                        "text": text,
                        "reply_markup": lead_actions_keyboard(str(lead.id)).to_dict(),
                    },
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(
                    "Telegram notification failed; lead remains unnotified",
                    extra={"lead_id": str(lead.id), "error": _redact_token(str(exc))},
                )
                continue
            lead.notified_at = datetime.utcnow()


def _redact_token(value: str) -> str:
    if not settings.telegram_bot_token:
        return value
    return value.replace(settings.telegram_bot_token, "<telegram-token>")
