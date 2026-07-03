from dataclasses import dataclass
from functools import lru_cache
import re

from app.core.config import get_settings
from app.normalizers.region import geography_priority_weight
from app.normalizers.text import normalize_text
from app.scoring.rules import (
    ACTION_KEYWORDS,
    GENERIC_EXHIBITION_KEYWORDS,
    HIGH_INTENT_KEYWORDS,
    LOW_RELEVANCE_KEYWORDS,
    SERVICE_ONLY_KEYWORDS,
    STRONG_BUYING_SIGNAL_KEYWORDS,
)


@lru_cache
def _get_scoring_settings() -> dict[str, int | list[str]]:
    settings = get_settings()
    whitelist = [normalize_text(item) for item in settings.scoring_whitelist_keywords if item.strip()]
    blacklist = [normalize_text(item) for item in settings.scoring_blacklist_keywords if item.strip()]
    return {
        "whitelist": whitelist,
        "blacklist": blacklist,
        "priority_a_threshold": settings.scoring_priority_a_threshold,
        "priority_b_threshold": settings.scoring_priority_b_threshold,
        "whitelist_weight": settings.scoring_whitelist_weight,
        "whitelist_cap": settings.scoring_whitelist_cap,
        "blacklist_weight": settings.scoring_blacklist_weight,
        "blacklist_cap": settings.scoring_blacklist_cap,
    }


@dataclass(frozen=True)
class ScoreDetails:
    score: int
    priority: str
    factors: list[str]


def _priority_from_score(score: int, threshold_a: int, threshold_b: int) -> str:
    if score >= threshold_a:
        return "A"
    if score >= threshold_b:
        return "B"
    return "C"


def _has_profile_order_signal(text: str) -> bool:
    patterns = (
        r"выставочн\w*\s+стенд",
        r"стенд\w*\s+выставочн",
        r"застройк\w*\s+.*стенд",
        r"создан\w*\s+.*стенд",
        r"монтаж\w*\s+.*стенд",
        r"демонтаж\w*\s+.*стенд",
        r"оформлен\w*\s+.*стенд",
        r"оформлен\w*\s+.*выставочн\w*\s+экспозиц",
        r"экспозиционн\w*\s+стенд",
        r"выставочн\w*\s+оборудован",
        r"брендирован\w*\s+.*стенд",
        r"регистрационн\w*\s+стойк",
        r"фотозон",
        r"промостенд",
        r"\bposm\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def explain_score_lead(
    title: str,
    description: str | None,
    region: str | None,
    budget_max: float | None,
) -> ScoreDetails:
    settings = _get_scoring_settings()
    score = 0
    factors: list[str] = []
    title_text = normalize_text(title)
    description_text = normalize_text(description or "")
    text = normalize_text(f"{title} {description or ''}")
    region_text = normalize_text(region or "")
    whitelist_keywords = settings["whitelist"]
    blacklist_keywords = settings["blacklist"]

    strong_signal_hits = sum(1 for keyword in STRONG_BUYING_SIGNAL_KEYWORDS if keyword in text)
    strong_title_signal_hits = sum(1 for keyword in STRONG_BUYING_SIGNAL_KEYWORDS if keyword in title_text)
    generic_exhibition_hits = sum(1 for keyword in GENERIC_EXHIBITION_KEYWORDS if keyword in text)
    action_hits = sum(1 for keyword in ACTION_KEYWORDS if keyword in text)
    high_intent_hits = sum(1 for keyword in HIGH_INTENT_KEYWORDS if keyword in text)
    low_relevance_hits = sum(1 for keyword in LOW_RELEVANCE_KEYWORDS if keyword in text)
    service_only_hits = sum(1 for keyword in SERVICE_ONLY_KEYWORDS if keyword in text)
    whitelist_hits = sum(1 for keyword in whitelist_keywords if keyword and keyword in text)
    blacklist_hits = sum(1 for keyword in blacklist_keywords if keyword and keyword in text)

    if _has_profile_order_signal(text):
        score += 30
        factors.append("Профильный заказ на стенд/застройку/экспозицию: +30")

    if _has_profile_order_signal(title_text):
        score += 12
        factors.append("Профильный сигнал есть прямо в заголовке: +12")

    region_bonus = geography_priority_weight(region_text)
    score += region_bonus
    if region_bonus:
        factors.append(f"География с учетом удаленности от Москвы: +{region_bonus}")

    strong_signal_bonus = min(strong_signal_hits * 18, 36)
    score += strong_signal_bonus
    if strong_signal_bonus:
        factors.append(f"Есть прямые признаки запроса на стенд/застройку: +{strong_signal_bonus}")

    title_signal_bonus = min(strong_title_signal_hits * 10, 20)
    score += title_signal_bonus
    if title_signal_bonus:
        factors.append(f"Сильные ключи есть прямо в заголовке: +{title_signal_bonus}")

    action_bonus = min(action_hits * 8, 16)
    score += action_bonus
    if action_bonus:
        factors.append(f"Есть признаки изготовления/монтажа/оформления: +{action_bonus}")

    high_intent_bonus = min(high_intent_hits * 5, 15)
    score += high_intent_bonus
    if high_intent_bonus:
        factors.append(f"Коммерческий сигнал по услугам дизайна/печати: +{high_intent_bonus}")

    if "международная выставка" in description_text:
        score += 4
        factors.append("Международная выставка: +4")

    if "заказ экспоместа" in text or "участие в выставке" in text:
        score += 6
        factors.append("Есть явный признак участия в выставке: +6")

    if generic_exhibition_hits and strong_signal_hits:
        score += 8
        factors.append("Выставочная тематика подтверждена профильными сигналами: +8")
    elif generic_exhibition_hits:
        score += 4
        factors.append("Есть общая выставочная тематика без явного подрядного запроса: +4")

    low_relevance_penalty = min(low_relevance_hits * 6, 24)
    score -= low_relevance_penalty
    if low_relevance_penalty:
        factors.append(f"Низкоприоритетная отраслевая тематика: -{low_relevance_penalty}")

    service_only_penalty = min(service_only_hits * 12, 24)
    score -= service_only_penalty
    if service_only_penalty:
        factors.append(f"Похоже на непрофильную сервисную задачу: -{service_only_penalty}")

    if generic_exhibition_hits and not strong_signal_hits and not action_hits:
        score -= 6
        factors.append("Похоже на отраслевую выставку без явных признаков заказа стенда: -6")

    whitelist_bonus = min(
        whitelist_hits * int(settings["whitelist_weight"]),
        int(settings["whitelist_cap"]),
    )
    score += whitelist_bonus
    if whitelist_bonus:
        factors.append(f"Whitelist-совпадения: +{whitelist_bonus}")

    blacklist_penalty = min(
        blacklist_hits * int(settings["blacklist_weight"]),
        int(settings["blacklist_cap"]),
    )
    score -= blacklist_penalty
    if blacklist_penalty:
        factors.append(f"Blacklist-совпадения: -{blacklist_penalty}")

    if budget_max and budget_max >= 1_000_000:
        score += 18
        factors.append("Бюджет от 1 000 000: +18")
    elif budget_max and budget_max >= 500_000:
        score += 12
        factors.append("Бюджет от 500 000: +12")
    elif budget_max and budget_max >= 300_000:
        score += 8
        factors.append("Бюджет от 300 000: +8")
    elif budget_max and budget_max < 100_000:
        score -= 15
        factors.append("Бюджет ниже 100 000: -15")

    priority = _priority_from_score(
        score,
        int(settings["priority_a_threshold"]),
        int(settings["priority_b_threshold"]),
    )
    return ScoreDetails(score=score, priority=priority, factors=factors)


def score_lead(title: str, description: str | None, region: str | None, budget_max: float | None) -> tuple[int, str]:
    details = explain_score_lead(title=title, description=description, region=region, budget_max=budget_max)
    return details.score, details.priority
