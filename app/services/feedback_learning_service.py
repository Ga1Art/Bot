from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Lead, LeadActivity
from app.schemas.collector import LeadCreate


ACCEPTED_ACTIONS = {"in_work", "accepted", "contacted", "won", "good_profile", "good_budget", "good_urgent"}
REJECTED_ACTIONS = {
    "rejected",
    "reject_not_profile",
    "reject_far",
    "reject_budget",
    "reject_deadline",
    "reject_duplicate",
    "reject_other",
}


@dataclass(frozen=True)
class LearningAdjustment:
    value: int
    reason: str | None = None


class FeedbackLearningService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    def score_adjustment_for(self, item: LeadCreate) -> LearningAdjustment:
        if not self.settings.enable_feedback_learning:
            return LearningAdjustment(0)

        rows = self._recent_feedback_leads()
        if not rows:
            return LearningAdjustment(0)

        accepted = 0
        rejected = 0
        reasons: list[str] = []
        item_tokens = self._tokens(item.title, item.description or "", item.keywords_matched or "")

        for lead, action_type in rows:
            weight = self._similarity_weight(item, item_tokens, lead)
            if weight <= 0:
                continue
            if action_type in ACCEPTED_ACTIONS:
                accepted += weight
            elif action_type in REJECTED_ACTIONS:
                rejected += weight

        raw_value = (accepted - rejected) * self.settings.feedback_learning_weight
        cap = max(0, self.settings.feedback_learning_cap)
        value = max(-cap, min(cap, raw_value))
        if value == 0:
            return LearningAdjustment(0)

        direction = "похожие лиды чаще принимали" if value > 0 else "похожие лиды чаще отклоняли"
        reasons.append(f"Learning: {direction} ({accepted}:{rejected})")
        return LearningAdjustment(value, "; ".join(reasons))

    def _recent_feedback_leads(self) -> list[tuple[Lead, str]]:
        stmt = (
            select(Lead, LeadActivity.action_type)
            .join(LeadActivity, LeadActivity.lead_id == Lead.id)
            .where(LeadActivity.action_type.in_(tuple(ACCEPTED_ACTIONS | REJECTED_ACTIONS)))
            .order_by(LeadActivity.created_at.desc())
            .limit(500)
        )
        return list(self.db.execute(stmt).all())

    def _similarity_weight(self, item: LeadCreate, item_tokens: set[str], lead: Lead) -> int:
        weight = 0
        if lead.source_name == item.source_name:
            weight += 1
        if lead.region and item.region and lead.region.lower() == item.region.lower():
            weight += 1
        if lead.customer_name and item.customer_name and lead.customer_name.lower() == item.customer_name.lower():
            weight += 2

        lead_tokens = self._tokens(lead.title, lead.description or "", lead.keywords_matched or "")
        overlap = item_tokens & lead_tokens
        if len(overlap) >= 2:
            weight += 1
        if len(overlap) >= 5:
            weight += 1
        return min(weight, 4)

    def _tokens(self, *values: str) -> set[str]:
        text = " ".join(values).lower().replace("ё", "е")
        raw_tokens = []
        token = []
        for char in text:
            if char.isalnum():
                token.append(char)
            elif token:
                raw_tokens.append("".join(token))
                token = []
        if token:
            raw_tokens.append("".join(token))
        return {token for token in raw_tokens if len(token) >= 4}
