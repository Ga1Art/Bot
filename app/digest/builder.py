from sqlalchemy.orm import Session

from app.bot.templates import format_money
from app.repositories.lead_repo import LeadRepository


def _shorten(value: str, limit: int = 120) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def build_digest_text(db: Session) -> str:
    repo = LeadRepository(db)
    leads = repo.list_latest_collected_new_leads()
    if not leads:
        return "Новых лидов из утреннего сбора нет."

    lines = [f"Новые лиды из утреннего сбора: {len(leads)}"]
    for index, lead in enumerate(leads[:10], start=1):
        deadline = lead.deadline_at.strftime("%d.%m.%Y %H:%M") if lead.deadline_at else "-"
        budget = format_money(lead.budget_max) if lead.budget_max is not None else "бюджет не указан"
        region = lead.region or "-"
        lines.append(f"{index}. [{lead.priority}] {_shorten(lead.title)}")
        lines.append(f"   {lead.source_name}, {region}, дедлайн {deadline}, {budget}")
        lines.append(f"   {lead.url}")
    if len(leads) > 10:
        lines.append(f"Показано 10 из {len(leads)}. Чтобы посмотреть все, нажми кнопку Новые или отправь /new.")
    return "\n".join(lines)
