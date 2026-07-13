from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.core.time import moscow_now_naive, moscow_tomorrow_start_naive
from app.normalizers.region import geography_priority_weight
from app.normalizers.text import normalize_text
from app.scoring.engine import explain_score_lead


@dataclass(frozen=True)
class ScoreComponents:
    base_score: int
    fit_score: int
    business_score: int
    urgency_score: int
    logistics_score: int
    quality_reason: str


def score_components(
    title: str,
    description: str | None,
    region: str | None,
    budget_max: float | None,
    deadline_at: datetime | None,
) -> ScoreComponents:
    details = explain_score_lead(
        title=title,
        description=description,
        region=region,
        budget_max=budget_max,
    )
    text = normalize_text(f"{title} {description or ''}")
    fit_score = _fit_score(text)
    business_score = _business_score(budget_max)
    logistics_score = geography_priority_weight(normalize_text(region or ""))
    urgency_score = _urgency_score(deadline_at)
    return ScoreComponents(
        base_score=details.score,
        fit_score=fit_score,
        business_score=business_score,
        urgency_score=urgency_score,
        logistics_score=logistics_score,
        quality_reason="; ".join(details.factors[:8]),
    )


def _fit_score(text: str) -> int:
    score = 0
    strong_patterns = (
        "выставочный стенд",
        "застройка стенда",
        "монтаж стенда",
        "оформление стенда",
        "экспозиционный стенд",
        "выставочное оборудование",
    )
    adjacent_patterns = (
        "брендирование",
        "фотозона",
        "регистрационная стойка",
        "posm",
        "рекламная конструкция",
    )
    score += min(sum(1 for item in strong_patterns if item in text) * 25, 75)
    score += min(sum(1 for item in adjacent_patterns if item in text) * 10, 25)
    if "тендерное сопровождение" in text or "вакансия" in text:
        score -= 30
    return max(0, min(100, score))


def _business_score(budget_max: float | None) -> int:
    if budget_max is None:
        return 35
    if budget_max >= 1_000_000:
        return 90
    if budget_max >= 500_000:
        return 75
    if budget_max >= 300_000:
        return 60
    if budget_max >= 100_000:
        return 35
    return 10


def _urgency_score(deadline_at: datetime | None) -> int:
    if deadline_at is None:
        return 25
    now = moscow_now_naive()
    if deadline_at < moscow_tomorrow_start_naive():
        return 0
    days_left = (deadline_at - now).total_seconds() / 86400
    if days_left <= 2:
        return 45
    if days_left <= 7:
        return 85
    if days_left <= 21:
        return 70
    return 50
