from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


class LeadRead(BaseModel):
    id: UUID
    source_type: str
    source_name: str
    title: str
    url: str
    city: str | None = None
    region: str | None = None
    budget_min: Decimal | None = None
    budget_max: Decimal | None = None
    customer_name: str | None = None
    priority: str
    status: str
    relevance_score: int
    base_relevance_score: int | None = None
    learned_score_adjustment: int = 0
    learned_reason: str | None = None
    ai_score: int | None = None
    ai_reason: str | None = None
    ai_recommended_action: str | None = None
    ai_tags: list[str] | None = None
    ai_risk_tags: list[str] | None = None
    ai_model: str | None = None
    ai_analyzed_at: datetime | None = None
    is_hot_prospect: bool = False
    published_at: datetime | None = None
    deadline_at: datetime | None = None

    class Config:
        from_attributes = True


class LeadStatusUpdate(BaseModel):
    status: str
    comment: str | None = None
    actor: str | None = None
