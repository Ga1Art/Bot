from datetime import timedelta
import re

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.time import moscow_tomorrow_start_naive
from app.collectors.tenders.common import has_closed_status
from app.db.models import CollectorRun, Lead
from app.normalizers.region import TARGET_EUROPEAN_RUSSIA_REGIONS
from app.schemas.collector import LeadCreate


COMMERCIAL_SOURCES_WITH_REQUIRED_DEADLINES = ("b2b_center", "bidzaar", "fabrikant", "rostender", "synapse")

LEAD_STRING_LIMITS = {
    "source_type": 32,
    "source_name": 128,
    "external_id": 256,
    "city": 128,
    "region": 128,
    "currency": 16,
    "customer_name": 256,
    "event_name": 256,
    "venue_name": 256,
    "priority": 8,
    "ai_recommended_action": 64,
    "ai_model": 128,
}


def _truncate_value(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _fingerprint(value: str) -> set[str]:
    text = value.lower().replace("ё", "е")
    return {token for token in re.findall(r"[a-zа-я0-9]{4,}", text) if token not in {"тендер", "закупка", "услуги"}}


class LeadRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _hot_leads_filters(self):
        settings = get_settings()
        target_region_filter = or_(
            Lead.region.in_(settings.priority_regions),
            *[Lead.region.ilike(f"%{region}%") for region in TARGET_EUROPEAN_RUSSIA_REGIONS],
        )
        return (
            Lead.status.in_(("new", "in_work")),
            Lead.priority.in_(("A", "B")),
            target_region_filter,
            self._active_deadline_filter(),
        )

    def _active_deadline_filter(self):
        return or_(Lead.deadline_at.is_(None), Lead.deadline_at >= moscow_tomorrow_start_naive())

    def list_leads(
        self,
        status: str | None = None,
        priority: str | None = None,
        region: str | None = None,
        source_type: str | None = None,
        hot_only: bool = False,
        limit: int | None = None,
    ) -> list[Lead]:
        stmt = select(Lead)
        if status:
            stmt = stmt.where(Lead.status == status)
            if status in ("new", "in_work"):
                stmt = stmt.where(self._active_deadline_filter())
        if priority:
            stmt = stmt.where(Lead.priority == priority)
        if region:
            stmt = stmt.where(Lead.region == region)
        if source_type:
            stmt = stmt.where(Lead.source_type == source_type)
        if hot_only:
            stmt = (
                stmt.where(*self._hot_leads_filters())
            )
        if hot_only:
            stmt = stmt.order_by(Lead.relevance_score.desc(), Lead.created_at.desc())
        else:
            stmt = stmt.order_by(Lead.created_at.desc())
        if limit:
            stmt = stmt.limit(limit)
        return list(self.db.scalars(stmt))

    def list_latest_collected_new_leads(self, window_minutes: int = 30) -> list[Lead]:
        latest_run = self.db.scalar(
            select(CollectorRun)
            .where(CollectorRun.status.in_(("success", "failed")))
            .where(CollectorRun.items_saved > 0)
            .order_by(CollectorRun.started_at.desc())
            .limit(1)
        )
        if latest_run is None:
            return []

        window_start = latest_run.started_at - timedelta(minutes=max(1, window_minutes))
        window_end = (latest_run.finished_at or latest_run.started_at) + timedelta(minutes=1)
        stmt = (
            select(Lead)
            .where(Lead.status == "new")
            .where(self._active_deadline_filter())
            .where(Lead.created_at >= window_start)
            .where(Lead.created_at <= window_end)
            .order_by(Lead.created_at.desc())
        )
        return list(self.db.scalars(stmt))

    def get_by_id(self, lead_id: str) -> Lead | None:
        return self.db.get(Lead, lead_id)

    def get_many_by_ids(self, lead_ids: list[str]) -> list[Lead]:
        if not lead_ids:
            return []

        stmt = select(Lead).where(Lead.id.in_(lead_ids))
        leads = list(self.db.scalars(stmt))
        order = {lead_id: index for index, lead_id in enumerate(lead_ids)}
        leads.sort(key=lambda item: order.get(str(item.id), len(order)))
        return leads

    def get_unnotified_priority_leads(self) -> list[Lead]:
        stmt = (
            select(Lead)
            .where(*self._hot_leads_filters())
            .where(Lead.notified_at.is_(None))
            .order_by(Lead.created_at.asc())
        )
        return list(self.db.scalars(stmt))

    def get_review_queue(self, limit: int = 20, offset: int = 0) -> list[Lead]:
        stmt = (
            select(Lead)
            .where(Lead.status.in_(("new", "in_work")))
            .where(self._active_deadline_filter())
            .order_by(Lead.priority.asc(), Lead.relevance_score.desc(), Lead.created_at.desc())
            .offset(max(0, offset))
            .limit(limit)
        )
        return list(self.db.scalars(stmt))

    def expire_stale_open_leads(self) -> int:
        stale_leads = list(
            self.db.scalars(
                select(Lead)
                .where(Lead.status == "new")
                .where(
                    or_(
                        Lead.deadline_at < moscow_tomorrow_start_naive(),
                        Lead.source_name.in_(COMMERCIAL_SOURCES_WITH_REQUIRED_DEADLINES) & Lead.deadline_at.is_(None),
                    )
                )
            )
        )
        stale_leads.extend(self._closed_status_open_leads())

        seen_ids: set[str] = set()
        unique_stale_leads: list[Lead] = []
        for lead in stale_leads:
            lead_id = str(lead.id)
            if lead_id in seen_ids:
                continue
            seen_ids.add(lead_id)
            unique_stale_leads.append(lead)

        for lead in unique_stale_leads:
            lead.status = "context"
        return len(unique_stale_leads)

    def _closed_status_open_leads(self) -> list[Lead]:
        commercial_open_leads = list(
            self.db.scalars(
                select(Lead)
                .where(Lead.status == "new")
                .where(Lead.source_name.in_(COMMERCIAL_SOURCES_WITH_REQUIRED_DEADLINES))
            )
        )
        closed_leads: list[Lead] = []
        for lead in commercial_open_leads:
            raw_block = ""
            if isinstance(lead.raw_payload, dict):
                raw_block = str(lead.raw_payload.get("block") or "")
            if has_closed_status(f"{lead.title or ''} {lead.description or ''} {raw_block}"):
                closed_leads.append(lead)
        return closed_leads

    def get_hot_prospects(self, limit: int = 20) -> list[Lead]:
        stmt = (
            select(Lead)
            .where(*self._hot_leads_filters())
            .order_by(Lead.relevance_score.desc(), Lead.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt))

    def get_by_source_external_id(self, source_name: str, external_id: str) -> Lead | None:
        stmt = select(Lead).where(
            Lead.source_name == source_name,
            Lead.external_id == external_id,
        )
        return self.db.scalar(stmt)

    def _find_potential_duplicate(self, payload: LeadCreate) -> Lead | None:
        if not payload.deadline_at:
            return None
        candidates = list(
            self.db.scalars(
                select(Lead)
                .where(Lead.source_name != payload.source_name)
                .where(Lead.deadline_at == payload.deadline_at)
                .where(Lead.status.in_(("new", "in_work")))
                .order_by(Lead.relevance_score.desc(), Lead.created_at.desc())
                .limit(50)
            )
        )
        if not candidates:
            return None

        payload_tokens = _fingerprint(f"{payload.title} {payload.description or ''}")
        if len(payload_tokens) < 3:
            return None
        for candidate in candidates:
            candidate_tokens = _fingerprint(f"{candidate.title} {candidate.description or ''}")
            overlap = payload_tokens & candidate_tokens
            overlap_ratio = len(overlap) / max(1, min(len(payload_tokens), len(candidate_tokens)))
            same_region = bool(payload.region and candidate.region and payload.region.lower() == candidate.region.lower())
            same_budget = bool(
                payload.budget_max is not None
                and candidate.budget_max is not None
                and abs(float(payload.budget_max) - float(candidate.budget_max)) < 1
            )
            if overlap_ratio >= 0.65 or (overlap_ratio >= 0.45 and (same_region or same_budget)):
                return candidate
        return None

    def upsert(
        self,
        payload: LeadCreate,
        relevance_score: int,
        priority: str,
        base_relevance_score: int | None = None,
        fit_score: int | None = None,
        business_score: int | None = None,
        urgency_score: int | None = None,
        logistics_score: int | None = None,
        quality_reason: str | None = None,
        learned_score_adjustment: int = 0,
        learned_reason: str | None = None,
        ai_score: int | None = None,
        ai_reason: str | None = None,
        ai_recommended_action: str | None = None,
        ai_tags: list[str] | None = None,
        ai_risk_tags: list[str] | None = None,
        ai_model: str | None = None,
        ai_analyzed_at=None,
    ) -> tuple[Lead, bool]:
        source_type = _truncate_value(payload.source_type, LEAD_STRING_LIMITS["source_type"]) or payload.source_type
        source_name = _truncate_value(payload.source_name, LEAD_STRING_LIMITS["source_name"]) or payload.source_name
        external_id = _truncate_value(payload.external_id, LEAD_STRING_LIMITS["external_id"]) or payload.external_id
        lead = self.get_by_source_external_id(source_name, external_id)
        created = lead is None
        city = _truncate_value(payload.city, LEAD_STRING_LIMITS["city"])
        region = _truncate_value(payload.region, LEAD_STRING_LIMITS["region"])
        currency = _truncate_value(payload.currency, LEAD_STRING_LIMITS["currency"])
        customer_name = _truncate_value(payload.customer_name, LEAD_STRING_LIMITS["customer_name"])
        event_name = _truncate_value(payload.event_name, LEAD_STRING_LIMITS["event_name"])
        venue_name = _truncate_value(payload.venue_name, LEAD_STRING_LIMITS["venue_name"])
        priority = _truncate_value(priority, LEAD_STRING_LIMITS["priority"]) or priority
        ai_recommended_action = _truncate_value(
            ai_recommended_action,
            LEAD_STRING_LIMITS["ai_recommended_action"],
        )
        ai_model = _truncate_value(ai_model, LEAD_STRING_LIMITS["ai_model"])

        if lead is None:
            duplicate = self._find_potential_duplicate(payload)
            lead = Lead(
                source_type=source_type,
                source_name=source_name,
                external_id=external_id,
                title=payload.title,
                description=payload.description,
                url=payload.url,
                published_at=payload.published_at,
                deadline_at=payload.deadline_at,
                city=city,
                region=region,
                budget_min=payload.budget_min,
                budget_max=payload.budget_max,
                currency=currency,
                customer_name=customer_name,
                event_name=event_name,
                venue_name=venue_name,
                keywords_matched=payload.keywords_matched,
                relevance_score=relevance_score,
                priority=priority,
                base_relevance_score=base_relevance_score,
                fit_score=fit_score,
                business_score=business_score,
                urgency_score=urgency_score,
                logistics_score=logistics_score,
                quality_reason=quality_reason,
                learned_score_adjustment=learned_score_adjustment,
                learned_reason=learned_reason,
                ai_score=ai_score,
                ai_reason=ai_reason,
                ai_recommended_action=ai_recommended_action,
                ai_tags=ai_tags,
                ai_risk_tags=ai_risk_tags,
                ai_model=ai_model,
                ai_analyzed_at=ai_analyzed_at,
                status="context" if duplicate else "new",
                is_duplicate=duplicate is not None,
                duplicate_of_id=duplicate.id if duplicate else None,
                duplicate_reason="Похожий заказ уже найден в другом источнике" if duplicate else None,
                raw_payload=payload.raw_payload,
            )
            self.db.add(lead)
            return lead, created

        lead.title = payload.title
        lead.description = payload.description
        lead.url = payload.url
        lead.published_at = payload.published_at
        lead.deadline_at = payload.deadline_at
        lead.city = city
        lead.region = region
        lead.budget_min = payload.budget_min
        lead.budget_max = payload.budget_max
        lead.currency = currency
        lead.customer_name = customer_name
        lead.event_name = event_name
        lead.venue_name = venue_name
        lead.keywords_matched = payload.keywords_matched
        lead.relevance_score = relevance_score
        lead.priority = priority
        lead.base_relevance_score = base_relevance_score
        lead.fit_score = fit_score
        lead.business_score = business_score
        lead.urgency_score = urgency_score
        lead.logistics_score = logistics_score
        lead.quality_reason = quality_reason
        lead.learned_score_adjustment = learned_score_adjustment
        lead.learned_reason = learned_reason
        if ai_score is not None or ai_analyzed_at is not None:
            lead.ai_score = ai_score
            lead.ai_reason = ai_reason
            lead.ai_recommended_action = ai_recommended_action
            lead.ai_tags = ai_tags
            lead.ai_risk_tags = ai_risk_tags
            lead.ai_model = ai_model
            lead.ai_analyzed_at = ai_analyzed_at
        lead.raw_payload = payload.raw_payload
        return lead, created

    def daily_counts(self) -> tuple[int, dict[str, int], dict[str, int]]:
        total = self.db.scalar(select(func.count()).select_from(Lead)) or 0
        priority_rows = self.db.execute(select(Lead.priority, func.count()).group_by(Lead.priority)).all()
        status_rows = self.db.execute(select(Lead.status, func.count()).group_by(Lead.status)).all()
        return total, dict(priority_rows), dict(status_rows)

    def hot_counts(
        self,
    ) -> tuple[
        int,
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        list[Lead],
    ]:
        base_stmt = select(Lead).where(*self._hot_leads_filters())

        total = self.db.scalar(select(func.count()).select_from(base_stmt.subquery())) or 0

        priority_rows = self.db.execute(
            select(Lead.priority, func.count())
            .where(*self._hot_leads_filters())
            .group_by(Lead.priority)
        ).all()

        status_rows = self.db.execute(
            select(Lead.status, func.count())
            .where(*self._hot_leads_filters())
            .group_by(Lead.status)
        ).all()

        region_rows = self.db.execute(
            select(Lead.region, func.count())
            .where(*self._hot_leads_filters())
            .group_by(Lead.region)
            .order_by(func.count().desc(), Lead.region.asc())
        ).all()

        source_rows = self.db.execute(
            select(Lead.source_name, func.count())
            .where(*self._hot_leads_filters())
            .group_by(Lead.source_name)
            .order_by(func.count().desc(), Lead.source_name.asc())
        ).all()

        customer_rows = self.db.execute(
            select(Lead.customer_name, func.count())
            .where(*self._hot_leads_filters())
            .where(Lead.customer_name.is_not(None))
            .where(Lead.customer_name != "")
            .group_by(Lead.customer_name)
            .order_by(func.count().desc(), Lead.customer_name.asc())
            .limit(10)
        ).all()

        event_rows = self.db.execute(
            select(Lead.event_name, func.count())
            .where(*self._hot_leads_filters())
            .where(Lead.event_name.is_not(None))
            .where(Lead.event_name != "")
            .group_by(Lead.event_name)
            .order_by(func.count().desc(), Lead.event_name.asc())
            .limit(10)
        ).all()

        deadline_rows = list(
            self.db.scalars(
                select(Lead)
                .where(*self._hot_leads_filters())
                .where(Lead.deadline_at.is_not(None))
                .order_by(Lead.deadline_at.asc(), Lead.relevance_score.desc(), Lead.created_at.desc())
                .limit(10)
            )
        )

        return (
            total,
            dict(priority_rows),
            dict(status_rows),
            dict(region_rows),
            dict(source_rows),
            dict(customer_rows),
            dict(event_rows),
            deadline_rows,
        )
