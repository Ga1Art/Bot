import logging

from sqlalchemy.orm import Session

from app.db.models import Lead
from app.repositories.collector_run_repo import CollectorRunRepository
from app.repositories.activity_repo import ActivityRepository
from app.repositories.lead_repo import LeadRepository
from app.schemas.collector import LeadCreate
from app.scoring.components import score_components
from app.core.config import get_settings
from app.services.feedback_learning_service import FeedbackLearningService
from app.services.telegram_service import TelegramService

logger = logging.getLogger(__name__)


class CollectorService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.leads = LeadRepository(db)
        self.activities = ActivityRepository(db)
        self.runs = CollectorRunRepository(db)
        self.telegram = TelegramService()
        self.feedback_learning = FeedbackLearningService(db)
        self.settings = get_settings()

    def ingest(self, items: list[LeadCreate]) -> tuple[int, int, list[Lead]]:
        found = len(items)
        saved = 0
        created_leads: list[Lead] = []

        for item in items:
            components = score_components(
                title=item.title,
                description=item.description,
                region=item.region,
                budget_max=float(item.budget_max) if item.budget_max is not None else None,
                deadline_at=item.deadline_at,
            )
            learning = self.feedback_learning.score_adjustment_for(item)
            score = components.base_score + learning.value
            priority = self._priority_from_score(score)
            lead, created = self.leads.upsert(
                item,
                relevance_score=score,
                priority=priority,
                base_relevance_score=components.base_score,
                fit_score=components.fit_score,
                business_score=components.business_score,
                urgency_score=components.urgency_score,
                logistics_score=components.logistics_score,
                quality_reason=components.quality_reason,
                learned_score_adjustment=learning.value,
                learned_reason=learning.reason,
            )
            if created:
                self.db.flush()
                self.activities.create(
                    lead_id=str(lead.id),
                    action_type="created",
                    comment=f"Imported from {item.source_name}",
                    actor="collector",
                )
                saved += 1
                created_leads.append(lead)

        self.db.commit()
        logger.info("Collector ingest finished", extra={"found": found, "saved": saved})
        return found, saved, created_leads

    def run_collector(self, collector_name: str, items: list[LeadCreate]) -> tuple[int, int]:
        run = self.runs.create_started(collector_name)
        self.db.commit()
        try:
            expired = self.leads.expire_stale_open_leads()
            if expired:
                logger.info("Expired stale open leads", extra={"expired": expired})
            found, saved, created_leads = self.ingest(items)
            self.runs.mark_finished(run, status="success", items_found=found, items_saved=saved)
            self.telegram.notify_new_priority_leads(created_leads)
            self.db.commit()
            return found, saved
        except Exception as exc:
            self.db.rollback()
            run = self.db.merge(run)
            self.runs.mark_finished(run, status="failed", items_found=0, items_saved=0, error_text=str(exc))
            self.db.commit()
            raise

    def _priority_from_score(self, score: int) -> str:
        if score >= self.settings.scoring_priority_a_threshold:
            return "A"
        if score >= self.settings.scoring_priority_b_threshold:
            return "B"
        return "C"
