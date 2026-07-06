from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Lead
from app.db.session import SessionLocal
from app.services.lead_service import LeadService


def main() -> None:
    settings = get_settings()
    if not settings.enable_ai_scoring or not (settings.gemini_api_key or settings.openai_api_key):
        print("AI scoring is disabled. Set ENABLE_AI_SCORING=true and GEMINI_API_KEY to use this script.")
        return

    with SessionLocal() as db:
        leads = list(
            db.scalars(
                select(Lead)
                .where(Lead.status.in_(("new", "in_work")))
                .where(Lead.ai_analyzed_at.is_(None))
                .order_by(Lead.relevance_score.desc(), Lead.created_at.desc())
                .limit(settings.ai_daily_analysis_limit)
            )
        )
        service = LeadService(db)
        analyzed = 0
        failed = 0
        for lead in leads:
            _, message = service.analyze_with_ai(str(lead.id))
            if message.startswith("AI-анализ обновлен"):
                analyzed += 1
            else:
                failed += 1
        print(f"AI analyzed={analyzed} failed={failed} selected={len(leads)}")


if __name__ == "__main__":
    main()
