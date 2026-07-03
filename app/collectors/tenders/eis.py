from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from app.collectors.base import BaseCollector
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


class EisCollector(BaseCollector):
    source_name = "eis"
    default_keywords = [
        "выставочный стенд",
        "изготовление выставочного стенда",
        "монтаж выставочного стенда",
        "застройка стенда",
        "оформление стенда",
        "оформление выставочной экспозиции",
        "брендирование стенда",
        "экспозиционный стенд",
        "выставочное оборудование",
        "регистрационная стойка",
        "фотозона",
    ]

    def __init__(self, keywords: list[str] | None = None) -> None:
        settings = get_settings()
        self.base_url = settings.eis_base_url
        self.keywords = keywords or self.default_keywords

    def fetch(self) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        session = requests.Session()
        session.trust_env = False
        headers = {
            "User-Agent": "Mozilla/5.0 LeadRadar/1.0",
            "Connection": "close",
        }
        for keyword in self.keywords:
            response = session.get(self._build_search_url(keyword), headers=headers, timeout=30)
            response.raise_for_status()
            self._normalize_response_encoding(response)
            items.extend(self.parse_html(response.text, keyword))
        return items

    def parse_html(self, html: str, keyword: str) -> list[LeadCreate]:
        soup = BeautifulSoup(html, "lxml")
        links_by_notice = self._extract_notice_links(soup)
        text = soup.get_text("\n", strip=True)
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        return self._parse_lines(lines, keyword, links_by_notice)

    def _parse_lines(
        self,
        lines: list[str],
        keyword: str,
        links_by_notice: dict[str, str],
    ) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        blocks = self._split_into_blocks(lines)
        for block in blocks:
            item = self._build_item(block, keyword, links_by_notice)
            if item:
                items.append(item)
        return items

    def _split_into_blocks(self, lines: list[str]) -> list[list[str]]:
        blocks: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            if line.startswith(("№ извещения", "Номер извещения")):
                if current:
                    blocks.append(current)
                current = [line]
            elif current:
                current.append(line)

        if current:
            blocks.append(current)

        return blocks

    def _build_item(
        self,
        block: list[str],
        keyword: str,
        links_by_notice: dict[str, str],
    ) -> LeadCreate | None:
        notice_number = self._extract_notice_number(block)
        title = self._extract_title(block)
        if not title:
            return None
        external_id = notice_number or self._make_external_id(title, keyword, block)

        region = self._extract_after_colon(block, ("Регион:",))
        customer_name = self._extract_after_colon(
            block,
            ("Заказчик:", "Организация, размещающая заказ:",),
        )
        budget = self._extract_money(block)
        deadline_raw = self._extract_after_colon(block, ("Окончание подачи заявок:",))
        url = links_by_notice.get(external_id) or self._extract_link(block) or self._build_search_url(keyword)

        return LeadCreate(
            source_type="tender",
            source_name=self.source_name,
            external_id=external_id,
            title=title,
            description=self._extract_description(block),
            url=url,
            deadline_at=self._parse_datetime(deadline_raw),
            city="Москва" if region == "Москва" else None,
            region=region,
            budget_max=budget,
            currency="RUB" if budget is not None else None,
            customer_name=customer_name,
            keywords_matched=keyword,
            raw_payload={
                "notice_number": notice_number,
                "search_keyword": keyword,
                "deadline_raw": deadline_raw,
                "block": block,
            },
        )

    def _extract_notice_number(self, block: list[str]) -> str | None:
        raw = self._extract_after_colon(block, ("№ извещения:", "Номер извещения:"))
        if raw:
            match = re.search(r"\d{6,}", raw)
            return match.group(0) if match else raw
        for line in block:
            match = re.search(r"\b\d{11,}\b", line)
            if match:
                return match.group(0)
        return None

    def _extract_title(self, block: list[str]) -> str | None:
        for idx, line in enumerate(block):
            if line.startswith(("Объект закупки:", "Наименование закупки:")):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
                if idx + 1 < len(block):
                    return block[idx + 1]
        return None

    def _extract_description(self, block: list[str]) -> str | None:
        description_lines: list[str] = []
        capture = False
        stop_prefixes = (
            "Начальная цена:",
            "НМЦК:",
            "Регион:",
            "Заказчик:",
            "Организация, размещающая заказ:",
            "Окончание подачи заявок:",
        )
        for line in block:
            if line.startswith("Описание объекта закупки:"):
                capture = True
                value = line.split(":", 1)[1].strip()
                if value:
                    description_lines.append(value)
                continue
            if capture:
                if line.startswith(stop_prefixes):
                    break
                description_lines.append(line)
        return " ".join(description_lines).strip() or None

    def _extract_after_colon(self, block: list[str], prefixes: tuple[str, ...]) -> str | None:
        for idx, line in enumerate(block):
            if line.startswith(prefixes):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
                if idx + 1 < len(block):
                    return block[idx + 1]
        return None

    def _extract_money(self, block: list[str]) -> Decimal | None:
        raw = self._extract_after_colon(block, ("Начальная цена:", "НМЦК:", "Цена:",))
        if not raw:
            return None
        normalized = re.sub(r"[^\d,\.]", "", raw).replace(",", ".")
        try:
            return Decimal(normalized)
        except Exception:
            return None

    def _extract_link(self, block: list[str]) -> str | None:
        for line in block:
            match = re.search(r"https?://\S+", line)
            if match:
                return match.group(0)
        return None

    def _extract_notice_links(self, soup: BeautifulSoup) -> dict[str, str]:
        links: dict[str, str] = {}
        for anchor in soup.select("a[href]"):
            href = anchor.get("href") or ""
            text = anchor.get_text(" ", strip=True)
            match = re.search(r"(?:regNumber=|/)(\d{11,})", href) or re.search(r"\b\d{11,}\b", text)
            if not match:
                continue
            notice = match.group(1)
            if href.startswith("/"):
                href = f"https://zakupki.gov.ru{href}"
            links[notice] = href
        return links

    def _build_search_url(self, keyword: str) -> str:
        encoded = quote_plus(keyword)
        return (
            f"{self.base_url}?searchString={encoded}&morphology=on&sortDirection=false"
            "&recordsPerPage=_10&showLotsInfoHidden=false&sortBy=UPDATE_DATE&pageNumber=1"
            "&fz44=on&fz223=on&af=on&currencyIdGeneral=-1"
        )

    def _make_external_id(self, title: str, keyword: str, block: list[str]) -> str:
        seed = f"{title}|{keyword}|{'|'.join(block[:4])}"
        return hashlib.md5(seed.encode("utf-8")).hexdigest()

    def _clean_line(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _normalize_response_encoding(self, response: requests.Response) -> None:
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        match = re.search(
            r"(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2}))?",
            value,
        )
        if not match:
            return None
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        hour = int(match.group(4) or 0)
        minute = int(match.group(5) or 0)
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None
