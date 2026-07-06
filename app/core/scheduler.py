from apscheduler.schedulers.background import BackgroundScheduler
import logging
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Lead
from app.db.session import SessionLocal
from app.digest.sender import send_digest
from app.services.lead_service import LeadService
from app.services.runner_service import RunnerService
from app.services.sync_service import SyncService

scheduler = BackgroundScheduler(timezone="Europe/Moscow")
logger = logging.getLogger(__name__)


def run_collectors_job() -> None:
    RunnerService().run_all()


def sync_google_sheets_job() -> None:
    SyncService().sync_queue_to_google_sheets()


def analyze_ai_leads_job() -> None:
    settings = get_settings()
    if not settings.enable_ai_scoring or not settings.openai_api_key:
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
        for lead in leads:
            try:
                service.analyze_with_ai(str(lead.id))
            except Exception as exc:
                logger.warning("Scheduled AI analysis failed", extra={"lead_id": str(lead.id), "error": str(exc)[:300]})


scheduler.add_job(run_collectors_job, "cron", minute="15", id="collectors_cycle")
scheduler.add_job(sync_google_sheets_job, "cron", minute="25", id="google_sheets_sync")
settings = get_settings()
scheduler.add_job(
    analyze_ai_leads_job,
    "cron",
    hour=settings.ai_analysis_hour,
    minute=settings.ai_analysis_minute,
    id="ai_leads_analysis",
)
scheduler.add_job(send_digest, "cron", hour=9, minute=0, id="morning_digest")
scheduler.add_job(send_digest, "cron", hour=16, minute=0, id="evening_digest")
