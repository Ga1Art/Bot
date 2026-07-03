from datetime import datetime

from pydantic import BaseModel


class DailyStats(BaseModel):
    total: int
    by_priority: dict[str, int]
    by_status: dict[str, int]


class HotDeadlineItem(BaseModel):
    title: str
    deadline_at: datetime
    priority: str
    region: str | None = None
    source_name: str
    customer_name: str | None = None
    url: str
    relevance_score: int


class HotLeadStats(BaseModel):
    total: int
    by_priority: dict[str, int]
    by_status: dict[str, int]
    by_region: dict[str, int]
    by_source: dict[str, int]
    top_customers: dict[str, int]
    top_events: dict[str, int]
    top_deadlines: list[HotDeadlineItem]
