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
    is_hot_prospect: bool = False
    published_at: datetime | None = None
    deadline_at: datetime | None = None

    class Config:
        from_attributes = True


class LeadStatusUpdate(BaseModel):
    status: str
    comment: str | None = None
    actor: str | None = None
