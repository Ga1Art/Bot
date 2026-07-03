from __future__ import annotations

import hashlib
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from app.schemas.collector import LeadCreate


class ExpocentrCollector:
    source_name = "expocentr"
    source_url = "https://www.expocentr.ru/ru/events/"

    def fetch(self) -> list[LeadCreate]:
        session = requests.Session()
        session.trust_env = False
        response = session.get(self.source_url, timeout=30)
        response.raise_for_status()
        self._normalize_response_encoding(response)
        return self.parse_html(response.text)

    def parse_html(self, html: str) -> list[LeadCreate]:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        lines = [self._clean_line(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        return self._parse_lines(lines)

    def _parse_lines(self, lines: list[str]) -> list[LeadCreate]:
        events: list[LeadCreate] = []
        i = 0
        while i < len(lines):
            if not self._is_date_range(lines[i]):
                i += 1
                continue

            date_range = lines[i]
            event_line = self._find_event_line(lines, i + 1)
            if not event_line:
                i += 1
                continue

            event = self._parse_event_line(event_line, date_range)
            if event:
                events.append(event)
            i += 1
        return events

    def _parse_event_line(self, line: str, date_range: str) -> LeadCreate | None:
        prefix = "Собственные выставки "
        if not line.startswith(prefix):
            return None

        payload = line[len(prefix) :].strip()
        venue_match = re.search(r"\s((?:МВЦ|ВДНХ|ВК)\s+.+?)\s+\d+\+$", payload)
        if not venue_match:
            return None

        venue = venue_match.group(1).strip()
        head = payload[: venue_match.start()].strip()
        title, description = self._split_title_and_description(head)
        deadline_at = self._extract_deadline(date_range)
        external_id = self._make_external_id(title, date_range or line)
        city, region = self._infer_location(venue)

        return LeadCreate(
            source_type="event",
            source_name=self.source_name,
            external_id=external_id,
            title=title,
            description=description,
            url=self.source_url,
            deadline_at=deadline_at,
            city=city,
            region=region,
            customer_name='АО "ЭКСПОЦЕНТР"',
            event_name=title,
            venue_name=venue,
            raw_payload={
                "line": line,
                "date_range": date_range,
                "venue": venue,
                "source_url": self.source_url,
            },
        )

    def _extract_deadline(self, date_range: str | None) -> datetime | None:
        if not date_range:
            return None
        end_raw = date_range.split("—", maxsplit=1)[1]
        try:
            return datetime.strptime(end_raw, "%d.%m.%Y")
        except ValueError:
            return None

    def _infer_location(self, venue: str) -> tuple[str | None, str | None]:
        if "Москва" in venue or "ВДНХ" in venue or "Сокольнический" in venue:
            return "Москва", "Москва"
        return None, None

    def _make_external_id(self, title: str, seed: str) -> str:
        digest = hashlib.md5(f"{title}|{seed}".encode("utf-8")).hexdigest()
        return digest

    def _find_event_line(self, lines: list[str], start_index: int) -> str | None:
        for idx in range(max(start_index - 6, 0), start_index):
            if "Собственные выставки" in lines[idx]:
                return lines[idx]
        for idx in range(start_index, min(start_index + 6, len(lines))):
            if "Собственные выставки" in lines[idx]:
                return lines[idx]
        return None

    def _is_date_range(self, value: str) -> bool:
        return bool(re.match(r"\d{2}\.\d{2}\.\d{4}—\d{2}\.\d{2}\.\d{4}$", value))

    def _split_title_and_description(self, value: str) -> tuple[str, str | None]:
        markers = [
            r"\d+-я\s",
            r"\d+-й\s",
            r"\d+-е\s",
            r"Международн",
            r"международн",
            r"Форум",
            r"форум",
            r"Выставка",
            r"выставка",
        ]
        positions = []
        for marker in markers:
            match = re.search(marker, value)
            if match:
                positions.append(match.start())

        if not positions:
            return value.strip(), None

        split_at = min(pos for pos in positions if pos > 0)
        title = value[:split_at].strip()
        description = value[split_at:].strip() or None
        return title, description

    def _normalize_response_encoding(self, response: requests.Response) -> None:
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"

    def _clean_line(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()
