from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import quote

from bs4 import BeautifulSoup
from bs4.element import Tag

from app.collectors.base import BaseCollector
from app.collectors.tenders.common import (
    DEFAULT_HEADERS,
    clean_text,
    create_session,
    first_regex_group,
    full_url,
    has_actionable_deadline,
    has_closed_status,
    is_profile_order,
    normalize_response_encoding,
    parse_money,
    relevant_to_keywords,
    stable_id,
)
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


class SynapseCollector(BaseCollector):
    source_name = "synapse"
    base_url = "https://synapsenet.ru"
    search_path = "/search"

    def __init__(self, keywords: list[str] | None = None) -> None:
        settings = get_settings()
        self.keywords = keywords or settings.eis_search_keywords
        self.max_pages = max(1, settings.synapse_search_max_pages or settings.tender_search_max_pages)

    def fetch(self) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        session = create_session()
        for keyword in self.keywords:
            for page in range(1, self.max_pages + 1):
                response = session.get(
                    self._build_search_url(keyword, page),
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

        for anchor in soup.select('a.sp-tb-title[href*="/zakupki/"]'):
            title = clean_text(anchor.get_text(" ", strip=True))
            href = anchor.get("href") or ""
            block = self._find_result_block(anchor)
            if block is None:
                continue

            block_text = clean_text(block.get_text(" ", strip=True))
            if len(title) < 12 or not relevant_to_keywords(f"{title} {block_text}", keyword):
                continue
            if has_closed_status(block_text):
                continue
            if not is_profile_order(f"{title} {block_text}"):
                continue

            deadline_at = self._extract_deadline(block_text)
            if not has_actionable_deadline(deadline_at):
                continue

            budget = parse_money(block_text)
            items.append(
                LeadCreate(
                    source_type="tender",
                    source_name=self.source_name,
                    external_id=self._extract_external_id(href, block_text, title, keyword),
                    title=title,
                    description=block_text,
                    url=full_url(self.base_url, href),
                    published_at=self._extract_published_at(block_text),
                    deadline_at=deadline_at,
                    city=self._extract_city(title, block_text),
                    region=self._extract_region(title, block_text),
                    budget_max=budget,
                    currency="RUB" if budget is not None else None,
                    customer_name=self._extract_customer(block_text),
                    keywords_matched=keyword,
                    raw_payload={
                        "search_keyword": keyword,
                        "block": block_text,
                        "platform": self._extract_platform(block_text),
                    },
                )
            )

        return self._dedupe(items)

    def _build_search_url(self, keyword: str, page: int = 1) -> str:
        path = "/search" if page == 1 else "/search/tenderi-po-regionam"
        url = f"{self.base_url}{path}?query={quote(keyword)}"
        if page > 1:
            url = f"{url}&page={page}"
        return url

    def _find_result_block(self, anchor: Tag) -> Tag | None:
        block = anchor.find_parent(class_="sp-tender-block")
        if isinstance(block, Tag):
            return block
        node: Tag = anchor
        for _ in range(5):
            parent = node.parent
            if not isinstance(parent, Tag):
                return None
            if "Закупка" in clean_text(parent.get_text(" ", strip=True)):
                return parent
            node = parent
        return None

    def _extract_external_id(self, href: str, block_text: str, title: str, keyword: str) -> str:
        match = re.search(r"Закупка\s+([A-Za-zА-Яа-я0-9#_-]+)", block_text)
        if match:
            return match.group(1)
        parts = [part for part in href.strip("/").split("/") if part]
        return parts[1] if len(parts) > 1 else stable_id(self.source_name, title, keyword)

    def _extract_deadline(self, block_text: str) -> datetime | None:
        ranges = re.findall(
            r"(\d{1,2}:\d{2})\s*·\s*(\d{1,2}\.\d{1,2}\.\d{4})\s*[—-]\s*(\d{1,2}:\d{2})\s*·\s*(\d{1,2}\.\d{1,2}\.\d{4})",
            block_text,
        )
        if ranges:
            _, _, end_time, end_date = ranges[-1]
            return self._parse_ru_datetime(end_date, end_time)

        values = re.findall(r"(\d{1,2}:\d{2})\s*·\s*(\d{1,2}\.\d{1,2}\.\d{4})", block_text)
        if values:
            end_time, end_date = values[-1]
            return self._parse_ru_datetime(end_date, end_time)

        dates = re.findall(r"\d{1,2}\.\d{1,2}\.\d{4}", block_text)
        if dates:
            return self._parse_ru_datetime(dates[-1], "00:00")
        return None

    def _extract_published_at(self, block_text: str) -> datetime | None:
        values = re.findall(r"(\d{1,2}:\d{2})\s*·\s*(\d{1,2}\.\d{1,2}\.\d{4})", block_text)
        if values:
            start_time, start_date = values[0]
            return self._parse_ru_datetime(start_date, start_time)
        return None

    def _parse_ru_datetime(self, date_value: str, time_value: str) -> datetime | None:
        try:
            day, month, year = [int(part) for part in date_value.split(".")]
            hour, minute = [int(part) for part in time_value.split(":")]
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    def _extract_customer(self, block_text: str) -> str | None:
        return first_regex_group(
            r"заказчик\s+(.+?)(?:\s+текущая закупка|\s+прием заявок|\s+приём заявок|\s+площадка|$)",
            block_text,
            re.IGNORECASE,
        )

    def _extract_platform(self, block_text: str) -> str | None:
        return first_regex_group(
            r"площадка\s+(.+?)(?:\s+•|\s+способ отбора|\s+прием заявок|\s+приём заявок|$)",
            block_text,
            re.IGNORECASE,
        )

    def _extract_city(self, title: str, block_text: str) -> str | None:
        text = f"{title} {block_text}".lower()
        city_patterns = {
            "Москва": ("москва", "москов"),
            "Санкт-Петербург": ("санкт-петербург", "спб"),
            "Екатеринбург": ("екатеринбург",),
            "Казань": ("казань",),
            "Сочи": ("сочи",),
            "Краснодар": ("краснодар",),
        }
        for city, variants in city_patterns.items():
            if any(variant in text for variant in variants):
                return city
        return None

    def _extract_region(self, title: str, block_text: str) -> str | None:
        text = f"{title} {block_text}"
        return first_regex_group(
            r"(Москва|Московская обл|Санкт-Петербург|Ленинградская обл|Свердловская обл|Краснодарский край|Республика Татарстан|Башкортостан|Коми)",
            text,
        )

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
