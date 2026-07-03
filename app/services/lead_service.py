from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.normalizers.region import is_priority_region
from app.repositories.activity_repo import ActivityRepository
from app.repositories.lead_repo import LeadRepository
from app.schemas.lead import LeadRead, LeadStatusUpdate
from app.schemas.stats import DailyStats, HotDeadlineItem, HotLeadStats


class LeadService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.leads = LeadRepository(db)
        self.activities = ActivityRepository(db)
        self.settings = get_settings()

    def _to_lead_read(self, item) -> LeadRead:
        is_hot = (
            item.status in ("new", "in_work")
            and item.priority in ("A", "B")
            and is_priority_region(item.region)
        )
        payload = LeadRead.model_validate(item).model_copy(update={"is_hot_prospect": is_hot})
        return payload

    def list_leads(
        self,
        status: str | None = None,
        priority: str | None = None,
        region: str | None = None,
        source_type: str | None = None,
        hot_only: bool = False,
        limit: int | None = None,
    ) -> list[LeadRead]:
        return [
            self._to_lead_read(item)
            for item in self.leads.list_leads(status, priority, region, source_type, hot_only, limit)
        ]

    def update_status(self, lead_id: str, payload: LeadStatusUpdate) -> LeadRead:
        lead = self.leads.get_by_id(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")

        lead.status = payload.status
        self.activities.create(
            lead_id=lead_id,
            action_type=payload.status,
            comment=payload.comment,
            actor=payload.actor,
        )
        self.db.commit()
        self.db.refresh(lead)
        return self._to_lead_read(lead)

    def daily_stats(self) -> DailyStats:
        total, by_priority, by_status = self.leads.daily_counts()
        return DailyStats(total=total, by_priority=by_priority, by_status=by_status)

    def hot_stats(self) -> HotLeadStats:
        total, by_priority, by_status, by_region, by_source, top_customers, top_events, top_deadlines = (
            self.leads.hot_counts()
        )
        return HotLeadStats(
            total=total,
            by_priority=by_priority,
            by_status=by_status,
            by_region=by_region,
            by_source=by_source,
            top_customers=top_customers,
            top_events=top_events,
            top_deadlines=[
                HotDeadlineItem(
                    title=item.title,
                    deadline_at=item.deadline_at,
                    priority=item.priority,
                    region=item.region,
                    source_name=item.source_name,
                    customer_name=item.customer_name,
                    url=item.url,
                    relevance_score=item.relevance_score,
                )
                for item in top_deadlines
                if item.deadline_at is not None
            ],
        )

    def review_queue(self, limit: int = 20) -> list[LeadRead]:
        return [self._to_lead_read(item) for item in self.leads.get_review_queue(limit=limit)]

    def hot_prospects(self, limit: int = 20) -> list[LeadRead]:
        return [self._to_lead_read(item) for item in self.leads.get_hot_prospects(limit=limit)]

    def mine_leads(self, actor: str, limit: int = 20) -> list[LeadRead]:
        lead_ids = self.activities.get_latest_in_work_lead_ids_by_actor(actor=actor, limit=limit)
        return [self._to_lead_read(item) for item in self.leads.get_many_by_ids(lead_ids)]
