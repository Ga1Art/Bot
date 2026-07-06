from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Lead, LeadActivity


class ActivityRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, lead_id: str, action_type: str, comment: str | None, actor: str | None) -> LeadActivity:
        activity = LeadActivity(
            lead_id=lead_id,
            action_type=action_type,
            comment=comment,
            actor=actor,
        )
        self.db.add(activity)
        return activity

    def get_latest_in_work_lead_ids_by_actor(self, actor: str, limit: int = 20) -> list[str]:
        latest_activity = (
            select(
                LeadActivity.lead_id,
                func.max(LeadActivity.created_at).label("latest_created_at"),
            )
            .group_by(LeadActivity.lead_id)
            .subquery()
        )

        stmt = (
            select(LeadActivity.lead_id)
            .join(
                latest_activity,
                (LeadActivity.lead_id == latest_activity.c.lead_id)
                & (LeadActivity.created_at == latest_activity.c.latest_created_at),
            )
            .join(Lead, Lead.id == LeadActivity.lead_id)
            .where(Lead.status == "in_work")
            .where(LeadActivity.action_type.in_(("in_work", "accepted")))
            .where(LeadActivity.actor == actor)
            .order_by(Lead.updated_at.desc())
            .limit(limit)
        )
        return [str(item) for item in self.db.scalars(stmt)]
