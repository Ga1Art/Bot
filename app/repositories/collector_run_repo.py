from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import CollectorRun


class CollectorRunRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_started(self, collector_name: str) -> CollectorRun:
        run = CollectorRun(collector_name=collector_name, status="started")
        self.db.add(run)
        self.db.flush()
        return run

    def mark_finished(
        self,
        run: CollectorRun,
        status: str,
        items_found: int,
        items_saved: int,
        error_text: str | None = None,
    ) -> CollectorRun:
        run.status = status
        run.items_found = items_found
        run.items_saved = items_saved
        run.error_text = error_text
        run.finished_at = datetime.utcnow()
        return run
