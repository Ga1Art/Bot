from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


class LeadCreate(BaseModel):
    source_type: str
    source_name: str
    external_id: str
    title: str
    description: str | None = None
    url: str
    published_at: datetime | None = None
    deadline_at: datetime | None = None
    city: str | None = None
    region: str | None = None
    budget_min: Decimal | None = None
    budget_max: Decimal | None = None
    currency: str | None = None
    customer_name: str | None = None
    event_name: str | None = None
    venue_name: str | None = None
    keywords_matched: str | None = None
    raw_payload: dict | None = None
