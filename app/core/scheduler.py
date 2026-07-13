from apscheduler.schedulers.background import BackgroundScheduler
import logging
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Lead
from app.db.session import SessionLocal
from app.digest.sender import send_digest
from app.repositories.lead_repo import LeadRepository
from app.services.ai_scoring_service import AiScoringService
from app.services.lead_service import LeadService
from app.services.runner_service import RunnerService
from app.services.sync_service import SyncService

scheduler = BackgroundScheduler(timezone="Europe/Moscow")
logger = logging.getLogger(__name__)


def run_collectors_job() -> None:
    RunnerService().run_all()


def morning_collection_digest_job() -> None:
    expire_stale_leads_job()
    RunnerService().run_all()
    send_digest()


def sync_google_sheets_job() -> None:
    SyncService().sync_queue_to_google_sheets()


def expire_stale_leads_job() -> None:
    with SessionLocal() as db:
        expired = LeadRepository(db).expire_stale_open_leads()
        db.commit()
    if expired:
        logger.info("Expired stale queue leads", extra={"expired": expired})


def analyze_ai_leads_job() -> None:
    settings = get_settings()
    ai_scoring = AiScoringService()
    if not ai_scoring.is_enabled():
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


scheduler.add_job(expire_stale_leads_job, "cron", minute="5", id="expire_stale_leads")
scheduler.add_job(sync_google_sheets_job, "cron", minute="25", id="google_sheets_sync")
settings = get_settings()
if settings.enable_scheduled_morning_collection:
    scheduler.add_job(
        morning_collection_digest_job,
        "cron",
        hour=settings.digest_hour_morning,
        minute=0,
        id="morning_collection_digest",
    )
scheduler.add_job(
    analyze_ai_leads_job,
    "cron",
    hour=settings.ai_analysis_hour,
    minute=settings.ai_analysis_minute,
    id="ai_leads_analysis",
)
