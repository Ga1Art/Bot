from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.normalizers.region import is_priority_region
from app.repositories.activity_repo import ActivityRepository
from app.repositories.lead_repo import LeadRepository
from app.schemas.collector import LeadCreate
from app.schemas.lead import LeadRead, LeadStatusUpdate
from app.schemas.stats import DailyStats, HotDeadlineItem, HotLeadStats
from app.services.ai_scoring_service import AiScoringService


class LeadService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.leads = LeadRepository(db)
        self.activities = ActivityRepository(db)
        self.settings = get_settings()
        self.ai_scoring = AiScoringService()

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

    def latest_collected_new_leads(self) -> list[LeadRead]:
        return [
            self._to_lead_read(item)
            for item in self.leads.list_latest_collected_new_leads(
                window_minutes=self.settings.recent_collection_window_minutes,
            )
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

    def record_feedback(
        self,
        lead_id: str,
        action_type: str,
        status: str,
        actor: str | None = None,
        comment: str | None = None,
    ) -> LeadRead:
        lead = self.leads.get_by_id(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")

        lead.status = status
        self.activities.create(
            lead_id=lead_id,
            action_type=action_type,
            comment=comment,
            actor=actor,
        )
        self.db.commit()
        self.db.refresh(lead)
        return self._to_lead_read(lead)

    def analyze_with_ai(self, lead_id: str) -> tuple[LeadRead, str]:
        lead = self.leads.get_by_id(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        if not self.ai_scoring.is_enabled():
            return self._to_lead_read(lead), "AI-анализ не включен. Заполни GEMINI_API_KEY и ENABLE_AI_SCORING=true."

        payload = LeadCreate(
            source_type=lead.source_type,
            source_name=lead.source_name,
            external_id=lead.external_id,
            title=lead.title,
            description=lead.description,
            url=lead.url,
            published_at=lead.published_at,
            deadline_at=lead.deadline_at,
            city=lead.city,
            region=lead.region,
            budget_min=lead.budget_min,
            budget_max=lead.budget_max,
            currency=lead.currency,
            customer_name=lead.customer_name,
            event_name=lead.event_name,
            venue_name=lead.venue_name,
            keywords_matched=lead.keywords_matched,
            raw_payload=lead.raw_payload,
        )
        base_score = lead.base_relevance_score or lead.relevance_score
        assessment = self.ai_scoring.assess(payload, base_score=base_score)
        if assessment is None:
            return self._to_lead_read(lead), "AI-анализ не смог оценить лид. Основная логика не затронута."

        ai_adjustment = self.ai_scoring.score_adjustment(assessment)
        lead.ai_score = assessment.score
        lead.ai_reason = assessment.reason
        lead.ai_recommended_action = assessment.recommended_action
        lead.ai_tags = assessment.tags
        lead.ai_risk_tags = assessment.risk_tags
        lead.ai_model = assessment.model
        lead.ai_analyzed_at = assessment.analyzed_at
        lead.relevance_score = base_score + (lead.learned_score_adjustment or 0) + ai_adjustment
        lead.priority = self._priority_from_score(lead.relevance_score)
        self.activities.create(
            lead_id=lead_id,
            action_type="ai_analyzed",
            comment=f"AI score={assessment.score}; action={assessment.recommended_action}",
            actor="ai",
        )
        self.db.commit()
        self.db.refresh(lead)
        return self._to_lead_read(lead), "AI-анализ обновлен."

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

    def _priority_from_score(self, score: int) -> str:
        if score >= self.settings.scoring_priority_a_threshold:
            return "A"
        if score >= self.settings.scoring_priority_b_threshold:
            return "B"
        return "C"
