from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from app.core.time import moscow_now_naive, moscow_tomorrow_start_naive


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 LeadRadar/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.7",
    "Connection": "close",
}


def create_session() -> requests.Session:
    session = requests.Session()
    # The local environment can contain a broken proxy. Public sources work better without it.
    session.trust_env = False
    return session


def normalize_response_encoding(response: requests.Response) -> None:
    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding or "utf-8"


def clean_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def full_url(base_url: str, href: str) -> str:
    return urljoin(base_url, href)


def stable_id(*parts: str) -> str:
    seed = "|".join(part for part in parts if part)
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2}))?",
        value,
    )
    if not match:
        return None
    try:
        return datetime(
            int(match.group(3)),
            int(match.group(2)),
            int(match.group(1)),
            int(match.group(4) or 0),
            int(match.group(5) or 0),
        )
    except ValueError:
        return None


def is_future_deadline(value: datetime | None) -> bool:
    return value is None or value >= moscow_now_naive()


def has_future_deadline(value: datetime | None) -> bool:
    return value is not None and value >= moscow_now_naive()


def has_actionable_deadline(value: datetime | None) -> bool:
    return value is not None and value >= moscow_tomorrow_start_naive()


def has_closed_status(text: str) -> bool:
    text_lower = text.lower()
    closed_patterns = (
        "закончено",
        "завершено",
        "завершен",
        "завершён",
        "прием заявок завершен",
        "приём заявок завершен",
        "прием заявок завершён",
        "приём заявок завершён",
        "прием заявок окончен",
        "приём заявок окончен",
        "подача заявок окончена",
        "срок подачи заявок истек",
        "срок подачи заявок истёк",
        "подведение итогов",
        "подведены итоги",
        "определение победителя",
        "победитель определен",
        "победитель определён",
        "отменено",
        "отменен",
        "отменён",
        "архив",
        "в архиве",
        "не состоялась",
        "не состоялось",
    )
    return any(pattern in text_lower for pattern in closed_patterns)


def parse_money(value: str | None) -> Decimal | None:
    if not value:
        return None
    match = re.search(r"(\d[\d\s]*(?:[,.]\s*\d{2})?)\s*(?:RUB|руб|₽)", value, re.IGNORECASE)
    if not match:
        return None
    normalized = match.group(1).replace(" ", "").replace(",", ".")
    try:
        return Decimal(normalized)
    except Exception:
        return None


def closest_block(anchor: Tag, max_len: int = 1200) -> Tag:
    node: Tag = anchor
    fallback: Tag = anchor
    for _ in range(5):
        parent = node.parent
        if not isinstance(parent, Tag):
            break
        text = clean_text(parent.get_text(" ", strip=True))
        if text:
            fallback = parent
        if 80 <= len(text) <= max_len:
            return parent
        node = parent
    return fallback


def first_regex_group(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return clean_text(match.group(1)) if match else None


def relevant_to_keywords(text: str, keyword: str) -> bool:
    text_lower = text.lower()
    keyword_words = [word for word in re.split(r"\W+", keyword.lower()) if len(word) >= 4]
    return any(word in text_lower for word in keyword_words)


def is_profile_order(text: str) -> bool:
    text_lower = text.lower()
    positive_patterns = (
        r"выставочн\w*\s+стенд",
        r"стенд\w*\s+выставочн",
        r"застройк\w*\s+стенд",
        r"создан\w*\s+выставочн\w*\s+стенд",
        r"монтаж\w*\s+.*стенд",
        r"демонтаж\w*\s+.*стенд",
        r"оформл\w*\s+.*выставочн\w*\s+зон",
        r"оформл\w*\s+.*выставочн\w*\s+экспозиц",
        r"оформлен\w*\s+.*стенд",
        r"оформлен\w*\s+.*выставочн\w*\s+экспозиц",
        r"рекламн\w*\s+конструкц",
        r"экспозиционн\w*\s+стенд",
        r"выставочн\w*\s+оборудован",
        r"брендированн\w*\s+продукц",
        r"брендирован\w*\s+.*стенд",
        r"регистрационн\w*\s+стойк",
        r"информационн\w*\s+стенд",
        r"мобильн\w*\s+информационн\w*\s+стенд",
        r"фотозон",
        r"промостенд",
        r"\bposm\b",
        r"информационн\w*\s+знак",
        r"фасадн\w*\s+вывеск",
        r"табличк",
    )
    negative_patterns = (
        r"здание\s+нежил",
        r"квартир",
        r"земельн\w*\s+участ",
        r"лекарственн\w*\s+препарат",
        r"медицинск\w*\s+оборудован",
        r"сход-?развал",
        r"спортивн\w*\s+оборудован",
        r"телевизор",
    )
    return any(re.search(pattern, text_lower) for pattern in positive_patterns) and not any(
        re.search(pattern, text_lower) for pattern in negative_patterns
    )
