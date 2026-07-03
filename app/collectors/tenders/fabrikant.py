from __future__ import annotations

import re
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from app.collectors.base import BaseCollector
from app.collectors.tenders.common import (
    DEFAULT_HEADERS,
    clean_text,
    closest_block,
    create_session,
    first_regex_group,
    full_url,
    has_actionable_deadline,
    has_closed_status,
    is_profile_order,
    normalize_response_encoding,
    parse_datetime,
    parse_money,
    relevant_to_keywords,
    stable_id,
)
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


class FabrikantCollector(BaseCollector):
    source_name = "fabrikant"
    base_url = "https://www.fabrikant.ru"
    search_path = "/procedure/search"

    def __init__(self, keywords: list[str] | None = None) -> None:
        settings = get_settings()
        self.keywords = keywords or settings.eis_search_keywords

    def fetch(self) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        session = create_session()
        for keyword in self.keywords:
            response = session.get(
                self._build_search_url(keyword),
                headers=DEFAULT_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            normalize_response_encoding(response)
            items.extend(self.parse_html(response.text, keyword))
        return self._dedupe(items)

    def parse_html(self, html: str, keyword: str) -> list[LeadCreate]:
        soup = BeautifulSoup(html, "lxml")
        items: list[LeadCreate] = []

        for anchor in soup.select('a[href*="/procedure/"], a[href*="/trades/procedure/view/"]'):
            title = clean_text(anchor.get_text(" ", strip=True))
            href = anchor.get("href") or ""
            if len(title) < 12 or title.lower() in {"платит победитель", "бесплатное участие", "анализ цен"}:
                continue
            if not relevant_to_keywords(title, keyword):
                continue

            block = closest_block(anchor)
            block_text = clean_text(block.get_text(" ", strip=True))
            if has_closed_status(block_text):
                continue
            if not is_profile_order(f"{title} {block_text}"):
                continue
            budget = parse_money(block_text)
            deadline_at = self._extract_date_after("Дата окончания приёма заявок", block_text) or self._extract_date_after(
                "Дата окончания приема заявок",
                block_text,
            )
            if not has_actionable_deadline(deadline_at):
                continue
            items.append(
                LeadCreate(
                    source_type="tender",
                    source_name=self.source_name,
                    external_id=self._extract_external_id(href, block_text, title, keyword),
                    title=title,
                    description=block_text,
                    url=full_url(self.base_url, href),
                    published_at=self._extract_date_after("Дата публикации", block_text),
                    deadline_at=deadline_at,
                    region=self._extract_region(title, block_text),
                    city=self._extract_city(title, block_text),
                    budget_max=budget,
                    currency="RUB" if budget is not None else None,
                    customer_name=self._extract_customer(block_text),
                    keywords_matched=keyword,
                    raw_payload={"search_keyword": keyword, "block": block_text},
                )
            )

        return self._dedupe(items)

    def _build_search_url(self, keyword: str) -> str:
        return f"{self.base_url}{self.search_path}?{urlencode({'query': keyword})}"

    def _extract_external_id(self, href: str, block_text: str, title: str, keyword: str) -> str:
        match = re.search(r"/(\d{10,})$", href) or re.search(r"№\s*([\w-]+)", block_text)
        if match:
            return match.group(1)
        view_id = href.rstrip("/").split("/")[-1]
        return view_id if view_id else stable_id(self.source_name, title, keyword)

    def _extract_customer(self, block_text: str) -> str | None:
        return first_regex_group(
            r"Организатор\s+(.+?)(?:\s+Дата публикации|\s+Дата окончания|\s+\d{1,2}\.\d{1,2}\.\d{4}|$)",
            block_text,
        )

    def _extract_date_after(self, label: str, block_text: str):
        escaped = re.escape(label)
        value = first_regex_group(rf"{escaped}\s+(\d{{1,2}}\.\d{{1,2}}\.\d{{4}}(?:\s+\d{{1,2}}:\d{{2}})?)", block_text)
        return parse_datetime(value)

    def _extract_city(self, title: str, block_text: str) -> str | None:
        text = f"{title} {block_text}".lower()
        city_patterns = {
            "Москва": ("москва", "г. москва"),
            "Екатеринбург": ("екатеринбург",),
            "Санкт-Петербург": ("санкт-петербург", "спб"),
            "Казань": ("казань",),
            "Сочи": ("сочи",),
            "Краснодар": ("краснодар",),
        }
        for city, variants in city_patterns.items():
            if any(variant in text for variant in variants):
                return city
        return None

    def _extract_region(self, title: str, block_text: str) -> str | None:
        city = self._extract_city(title, block_text)
        if city == "Москва":
            return "Москва"
        if city == "Санкт-Петербург":
            return "Санкт-Петербург"
        if city == "Екатеринбург":
            return "Свердловская область"
        if city == "Казань":
            return "Республика Татарстан"
        if city in {"Сочи", "Краснодар"}:
            return "Краснодарский край"
        return None

    def _dedupe(self, items: list[LeadCreate]) -> list[LeadCreate]:
        seen: set[str] = set()
        unique: list[LeadCreate] = []
        for item in items:
            key = item.external_id
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique
