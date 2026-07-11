from __future__ import annotations

import re
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
    parse_datetime,
    parse_money,
    relevant_to_keywords,
    stable_id,
)
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


class RostenderCollector(BaseCollector):
    source_name = "rostender"
    base_url = "https://rostender.info"
    search_path = "/extsearch"

    def __init__(self, keywords: list[str] | None = None) -> None:
        settings = get_settings()
        self.keywords = keywords or settings.rostender_search_keywords or settings.eis_search_keywords
        self.max_pages = max(1, settings.rostender_search_max_pages or settings.tender_search_max_pages)

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

        for article in soup.select("article.tender-row"):
            block_text = clean_text(article.get_text(" ", strip=True))
            if has_closed_status(block_text):
                continue

            anchor = self._find_title_anchor(article)
            if anchor is None:
                continue

            title = clean_text(anchor.get_text(" ", strip=True))
            href = anchor.get("href") or ""
            if len(title) < 12 or not relevant_to_keywords(f"{title} {block_text}", keyword):
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
                    region=self._extract_region(article, title, block_text),
                    budget_max=budget,
                    currency="RUB" if budget is not None else None,
                    customer_name=self._extract_customer(block_text),
                    keywords_matched=keyword,
                    raw_payload={"search_keyword": keyword, "block": block_text},
                )
            )

        return self._dedupe(items)

    def _build_search_url(self, keyword: str, page: int = 1) -> str:
        url = f"{self.base_url}{self.search_path}?keywords={quote(keyword)}"
        if page > 1:
            url = f"{url}&page={page}"
        return url

    def _find_title_anchor(self, article: Tag) -> Tag | None:
        anchors = article.select('a[href*="-tender-"], a[href*="/tender-"]') or article.select("a[href]")
        for anchor in anchors:
            title = clean_text(anchor.get_text(" ", strip=True))
            href = anchor.get("href") or ""
            if len(title) >= 12 and "tender" in href:
                return anchor
        return None

    def _extract_external_id(self, href: str, block_text: str, title: str, keyword: str) -> str:
        match = re.search(r"/(\d{6,})-tender", href) or re.search(r"Тендер\s*№\s*(\d+)", block_text)
        return match.group(1) if match else stable_id(self.source_name, title, keyword)

    def _extract_deadline(self, block_text: str):
        value = first_regex_group(
            r"Окончание\s*\(МСК\)\s+(\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?)",
            block_text,
        )
        if value:
            return parse_datetime(value)

        dates = re.findall(r"\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2}", block_text)
        if dates:
            return parse_datetime(self._iso_to_ru_datetime(dates[-1]))

        ru_dates = re.findall(r"\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2})?", block_text)
        return parse_datetime(ru_dates[-1]) if ru_dates else None

    def _extract_published_at(self, block_text: str):
        value = first_regex_group(r"от\s+(\d{1,2}\.\d{1,2}\.\d{2,4})", block_text)
        if not value:
            return None
        if re.search(r"\d{1,2}\.\d{1,2}\.\d{2}$", value):
            value = f"{value[:-2]}20{value[-2:]}"
        return parse_datetime(value)

    def _extract_region(self, article: Tag, title: str, block_text: str) -> str | None:
        region_link = None
        for link in article.select('a[href*="/region/"]'):
            if "Закупки в регионе" in clean_text(link.get_text(" ", strip=True)):
                region_link = link
                break
        if region_link is not None:
            region = clean_text(region_link.get_text(" ", strip=True))
            region = re.sub(r"^Закупки в регионе\s+", "", region)
            if region:
                return region

        return first_regex_group(
            r"(Москва|Московская обл|Санкт-Петербург|Ленинградская обл|Свердловская обл|Краснодарский край|Республика Татарстан)",
            f"{title} {block_text}",
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

    def _extract_customer(self, block_text: str) -> str | None:
        customer = first_regex_group(
            r"(?:Заказчик|Организатор)\s+(.+?)(?:\s+Окончание|\s+Начальная цена|\s+\d{1,2}\.\d{1,2}\.\d{4}|$)",
            block_text,
        )
        if not customer or self._is_masked_customer(customer):
            return None
        return customer

    def _is_masked_customer(self, value: str) -> bool:
        text = clean_text(value)
        if not text:
            return True
        if re.search(r"[A-Za-zА-Яа-яЁё0-9]", text):
            return False
        return bool(re.search(r"[░█▓▒]{3,}", text))

    def _iso_to_ru_datetime(self, value: str) -> str:
        match = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}:\d{2})", value)
        if not match:
            return value
        return f"{match.group(3)}.{match.group(2)}.{match.group(1)} {match.group(4)}"

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
