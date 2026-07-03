from __future__ import annotations

import re
from urllib.parse import urlencode

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
    parse_datetime,
    parse_money,
    relevant_to_keywords,
    stable_id,
)
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


class B2BCenterCollector(BaseCollector):
    source_name = "b2b_center"
    base_url = "https://www.b2b-center.ru"
    search_path = "/market/"

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

        for anchor in soup.select('a[href*="tender-"]'):
            title = clean_text(anchor.get_text(" ", strip=True))
            href = anchor.get("href") or ""
            if len(title) < 12 or not relevant_to_keywords(title, keyword):
                continue

            row = anchor.find_parent("tr")
            block_text = clean_text(row.get_text(" ", strip=True)) if isinstance(row, Tag) else title
            if has_closed_status(block_text):
                continue
            if not is_profile_order(f"{title} {block_text}"):
                continue
            row_cells = self._row_cells(row)
            budget = parse_money(block_text)
            dates = re.findall(r"\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?", block_text)
            deadline_at = parse_datetime(dates[-1]) if len(dates) >= 2 else None
            if not has_actionable_deadline(deadline_at):
                continue

            items.append(
                LeadCreate(
                    source_type="tender",
                    source_name=self.source_name,
                    external_id=self._extract_external_id(href, title, keyword),
                    title=title,
                    description=block_text,
                    url=full_url(self.base_url, href),
                    published_at=parse_datetime(dates[0]) if dates else None,
                    deadline_at=deadline_at,
                    city=self._extract_city(title, block_text),
                    region=self._extract_region(title, block_text),
                    budget_max=budget,
                    currency="RUB" if budget is not None else None,
                    customer_name=self._extract_customer(block_text, title, row_cells),
                    keywords_matched=keyword,
                    raw_payload={"search_keyword": keyword, "block": block_text},
                )
            )

        return self._dedupe(items)

    def _build_search_url(self, keyword: str) -> str:
        return f"{self.base_url}{self.search_path}?{urlencode({'f_keyword': keyword, 'searching': '1'})}"

    def _extract_external_id(self, href: str, title: str, keyword: str) -> str:
        match = re.search(r"tender-(\d+)", href) or re.search(r"№\s*(\d+)", title)
        return match.group(1) if match else stable_id(self.source_name, title, keyword)

    def _row_cells(self, row: Tag | None) -> list[str]:
        if not isinstance(row, Tag):
            return []
        return [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all("td", recursive=False)]

    def _extract_customer(self, block_text: str, title: str, row_cells: list[str]) -> str | None:
        if len(row_cells) >= 3:
            customer = row_cells[2]
            if customer and not re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", customer):
                return customer
        tail = block_text.replace(title, " ", 1)
        dates = re.findall(r"\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?", tail)
        if dates:
            before_date = tail.split(dates[0], 1)[0]
            customer = clean_text(before_date)
            if customer:
                return customer
        return first_regex_group(r"(?:ООО|АО|ПАО|ИП|ФГБУ|ГБУ)\s+[^0-9]+", block_text)

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
