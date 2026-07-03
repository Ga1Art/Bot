from __future__ import annotations

import hashlib
import re
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from app.schemas.collector import LeadCreate


class ExponetCityCollector:
    def __init__(self, city_slug: str, city_name: str, region_name: str | None = None) -> None:
        self.city_slug = city_slug
        self.city_name = city_name
        self.region_name = region_name or city_name
        self.source_name = f"exponet_{city_slug}"
        self.source_url = (
            f"https://www.exponet.ru/exhibitions/countries/rus/cities/{city_slug}/dates/future/index.ru.html"
        )
        self.source_urls = [
            self.source_url,
            self.source_url.replace("https://", "http://"),
        ]

    def fetch(self) -> list[LeadCreate]:
        errors: list[str] = []
        session = requests.Session()
        session.trust_env = False
        headers = {"User-Agent": "Mozilla/5.0", "Connection": "close"}

        for url in self.source_urls:
            try:
                response = session.get(url, timeout=30, headers=headers, allow_redirects=True)
                response.raise_for_status()
                self._normalize_response_encoding(response)
                return self.parse_html(response.text)
            except Exception as exc:
                errors.append(f"{url}: {exc}")

        raise RuntimeError("; ".join(errors))

    def parse_html(self, html: str) -> list[LeadCreate]:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        detail_links = self._extract_detail_links(soup)
        return self._parse_lines(lines, detail_links)

    def _parse_lines(self, lines: list[str], detail_links: dict[str, str]) -> list[LeadCreate]:
        events: list[LeadCreate] = []
        i = 0

        while i < len(lines) - 2:
            if not self._is_date_range(lines[i], lines[i + 1]):
                i += 1
                continue

            date_range = f"{lines[i]}—{lines[i + 1]}"
            title = lines[i + 2]

            j = i + 3
            city = self.city_name
            region = self.region_name
            if j < len(lines) and re.match(r"^\(г\.\s*.+\)$", lines[j]):
                city = lines[j].replace("(г.", "").replace(")", "").strip()
                region = city if city != "Москва" else "Москва"
                j += 1

            description = lines[j] if j < len(lines) else None
            if description in {"Заказ экспоместа", "Получить приглашение"}:
                description = None

            deadline_at = self._extract_deadline(lines[i + 1])
            url = detail_links.get(title, self.source_url)
            external_id = self._make_external_id(title, date_range)

            events.append(
                LeadCreate(
                    source_type="event",
                    source_name=self.source_name,
                    external_id=external_id,
                    title=title,
                    description=description,
                    url=url,
                    deadline_at=deadline_at,
                    city=city,
                    region=region,
                    customer_name="Exponet",
                    event_name=title,
                    venue_name=city,
                    raw_payload={
                        "date_range": date_range,
                        "source_url": self.source_url,
                    },
                )
            )

            i = j + 1

        return events

    def _extract_detail_links(self, soup: BeautifulSoup) -> dict[str, str]:
        links: dict[str, str] = {}
        for a in soup.find_all("a", href=True):
            text = " ".join(a.get_text(" ", strip=True).split())
            href = a["href"]
            if not text or "/exhibitions/by-id/" not in href or "/participation." in href:
                continue
            links[text] = urljoin(self.source_url, href)
        return links

    def _is_date_range(self, start: str, end: str) -> bool:
        return bool(re.match(r"\d{2}\.\d{2}$", start) and re.match(r"\d{2}\.\d{2}\.\d{4}$", end))

    def _extract_deadline(self, end_raw: str) -> datetime | None:
        try:
            return datetime.strptime(end_raw, "%d.%m.%Y")
        except ValueError:
            return None

    def _make_external_id(self, title: str, seed: str) -> str:
        return hashlib.md5(f"{title}|{seed}".encode("utf-8")).hexdigest()

    def _normalize_response_encoding(self, response: requests.Response) -> None:
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "cp1251"

    def _clean_line(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()
