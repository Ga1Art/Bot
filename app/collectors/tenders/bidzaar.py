from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from app.collectors.base import BaseCollector
from app.collectors.tenders.common import (
    DEFAULT_HEADERS,
    clean_text,
    create_session,
    has_actionable_deadline,
    has_closed_status,
    is_profile_order,
)
from app.core.config import get_settings
from app.schemas.collector import LeadCreate


MOSCOW_TZ = ZoneInfo("Europe/Moscow")
UTC_TZ = ZoneInfo("UTC")


class BidzaarCollector(BaseCollector):
    source_name = "bidzaar"
    base_url = "https://bidzaar.com"
    api_url = "https://bidzaar.com/api/process/light/procedures/available"
    page_size = 25

    def __init__(self, keywords: list[str] | None = None) -> None:
        settings = get_settings()
        self.keywords = keywords or settings.eis_search_keywords
        self.max_pages = max(1, settings.bidzaar_search_max_pages or settings.tender_search_max_pages)

    def fetch(self) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        session = create_session()
        for keyword in self.keywords:
            for page in range(1, self.max_pages + 1):
                response = session.get(
                    self.api_url,
                    params=self._build_params(keyword, page),
                    headers={**DEFAULT_HEADERS, "Accept": "application/json"},
                    timeout=30,
                )
                response.raise_for_status()
                items.extend(self.parse_payload(response.json(), keyword))
        return self._dedupe(items)

    def parse_payload(self, payload: dict, keyword: str) -> list[LeadCreate]:
        items: list[LeadCreate] = []
        for raw_item in payload.get("items") or []:
            if not isinstance(raw_item, dict):
                continue

            # procedureType=1 is a buy request. Sales and registries are not actionable for us.
            if raw_item.get("procedureType") != 1:
                continue

            title = clean_text(str(raw_item.get("name") or ""))
            if len(title) < 12:
                continue

            addresses = raw_item.get("deliveryAddresses") or []
            address = addresses[0] if addresses and isinstance(addresses[0], dict) else {}
            description = self._description(raw_item, address)
            full_text = f"{title} {description}"
            if has_closed_status(full_text):
                continue
            if not is_profile_order(full_text):
                continue

            deadline_at = self._parse_iso_datetime(raw_item.get("acceptanceEndDate"))
            if not has_actionable_deadline(deadline_at):
                continue

            published_at = self._parse_iso_datetime(raw_item.get("publishDate"))
            external_id = str(raw_item.get("id") or raw_item.get("number") or "")
            if not external_id:
                continue

            city, region = self._extract_location(title, description, address)
            budget = self._extract_budget(raw_item)

            items.append(
                LeadCreate(
                    source_type="tender",
                    source_name=self.source_name,
                    external_id=external_id,
                    title=title,
                    description=description,
                    url=self._item_url(raw_item),
                    published_at=published_at,
                    deadline_at=deadline_at,
                    city=city,
                    region=region,
                    budget_max=budget,
                    currency="RUB" if budget is not None else None,
                    customer_name=clean_text(str(raw_item.get("companyName") or "")) or None,
                    keywords_matched=keyword,
                    raw_payload={"search_keyword": keyword, "item": raw_item, "block": full_text},
                )
            )

        return self._dedupe(items)

    def _build_params(self, keyword: str, page: int) -> dict[str, str]:
        return {
            "paging.page": str(page),
            "paging.size": str(self.page_size),
            "sorting.key": "publishDate",
            "sorting.direction": "desc",
            "logic": "and",
            "filters": "[]",
            "search": keyword,
        }

    def _description(self, item: dict, address: dict) -> str:
        parts = [
            f"Номер: {item.get('number')}" if item.get("number") else "",
            f"Заказчик: {item.get('companyName')}" if item.get("companyName") else "",
            f"Регион: {address.get('region')}" if address.get("region") else "",
            f"Город: {address.get('city')}" if address.get("city") else "",
            f"Адрес: {address.get('comment')}" if address.get("comment") else "",
        ]
        return clean_text(" ".join(part for part in parts if part))

    def _extract_location(self, title: str, description: str, address: dict) -> tuple[str | None, str | None]:
        city = clean_text(str(address.get("city") or "")) or None
        region = self._normalize_region(clean_text(str(address.get("region") or "")) or None)
        if city or region:
            return city, region or city

        text = f"{title} {description}".lower()
        city_regions = {
            "москва": ("Москва", "Москва"),
            "павловская слобода": ("Павловская Слобода", "Московская обл"),
            "санкт-петербург": ("Санкт-Петербург", "Санкт-Петербург"),
            "спб": ("Санкт-Петербург", "Санкт-Петербург"),
            "краснодар": ("Краснодар", "Краснодарский край"),
            "брянск": ("Брянск", "Брянская обл"),
            "нижний новгород": ("Нижний Новгород", "Нижегородская обл"),
            "таганрог": ("Таганрог", "Ростовская обл"),
            "ростов-на-дону": ("Ростов-на-Дону", "Ростовская обл"),
            "самара": ("Самара", "Самарская обл"),
            "чапаевск": ("Чапаевск", "Самарская обл"),
            "казань": ("Казань", "Республика Татарстан"),
            "волгоград": ("Волгоград", "Волгоградская обл"),
            "воронеж": ("Воронеж", "Воронежская обл"),
            "екатеринбург": ("Екатеринбург", "Свердловская обл"),
            "челябинск": ("Челябинск", "Челябинская обл"),
            "уфа": ("Уфа", "Республика Башкортостан"),
            "пермь": ("Пермь", "Пермский край"),
            "сочи": ("Сочи", "Краснодарский край"),
        }
        for needle, location in city_regions.items():
            if needle in text:
                return location
        return None, None

    def _normalize_region(self, region: str | None) -> str | None:
        if not region:
            return None
        normalized = region.strip()
        if normalized.lower() in {"г москва", "город москва"}:
            return "Москва"
        if normalized.lower() in {"г санкт-петербург", "город санкт-петербург"}:
            return "Санкт-Петербург"
        return normalized

    def _parse_iso_datetime(self, value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(MOSCOW_TZ).replace(tzinfo=None)

    def _extract_budget(self, item: dict) -> Decimal | None:
        for key in ("startPrice", "initialPrice", "maxPrice", "price", "budget"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return Decimal(str(value))
            except Exception:
                continue
        return None

    def _item_url(self, item: dict) -> str:
        number = str(item.get("number") or "")
        if number:
            return f"{self.base_url}/app/requests/public/buy?{urlencode({'search': number})}"
        return f"{self.base_url}/app/requests/public/buy"

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
