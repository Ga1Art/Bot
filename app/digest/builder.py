from sqlalchemy.orm import Session

from app.bot.templates import render_lead_card
from app.repositories.lead_repo import LeadRepository


def build_digest_text(db: Session) -> str:
    repo = LeadRepository(db)
    leads = repo.list_leads(status="new")
    if not leads:
        return "Новых лидов нет."

    lines = ["Новые лиды:"]
    for lead in leads[:10]:
        lines.append(
            render_lead_card(
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
        )
    return "\n".join(lines)
