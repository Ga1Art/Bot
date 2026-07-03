from apscheduler.schedulers.background import BackgroundScheduler

from app.digest.sender import send_digest
from app.services.runner_service import RunnerService
from app.services.sync_service import SyncService

scheduler = BackgroundScheduler(timezone="Europe/Moscow")


def run_collectors_job() -> None:
    RunnerService().run_all()


def sync_google_sheets_job() -> None:
    SyncService().sync_queue_to_google_sheets()


scheduler.add_job(run_collectors_job, "cron", minute="15", id="collectors_cycle")
scheduler.add_job(sync_google_sheets_job, "cron", minute="25", id="google_sheets_sync")
scheduler.add_job(send_digest, "cron", hour=9, minute=0, id="morning_digest")
scheduler.add_job(send_digest, "cron", hour=16, minute=0, id="evening_digest")
