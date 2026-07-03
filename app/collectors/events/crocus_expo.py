from __future__ import annotations

import hashlib
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from app.schemas.collector import LeadCreate


class CrocusExpoCollector:
    source_name = "crocus_expo"
    source_url = "http://crocus-expo.ru/exhibition/"
    source_urls = [
        "http://crocus-expo.ru/exhibition/",
        "http://www.crocus-expo.ru/exhibition/",
        "https://www.crocus-expo.ru/exhibition/",
    ]
    MONTHS = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }

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
        return self._parse_lines(lines)

    def _parse_lines(self, lines: list[str]) -> list[LeadCreate]:
        events: list[LeadCreate] = []
        i = 0

        while i < len(lines):
            if not self._is_date_range(lines[i]):
                i += 1
                continue

            title, description = self._extract_title_and_description(lines, i)
            start = i
            end = self._find_block_end(lines, i + 1)
            block = lines[start:end]
            event = self._build_event(title=title, description=description, block=block)
            if event:
                events.append(event)
            i = end

        return events

    def _build_event(self, title: str | None, description: str | None, block: list[str]) -> LeadCreate | None:
        if not title:
            return None

        deadline_at = self._extract_deadline(block[0])
        organizer = self._extract_field(block, "Организатор:")
        website = self._extract_website(block)
        summary = self._extract_summary(block)
        venue = self._extract_field(block, "Место проведения:")

        description_parts = [part for part in [description, summary] if part]
        final_description = "\n\n".join(description_parts) if description_parts else None
        external_id = self._make_external_id(title, block[0])

        return LeadCreate(
            source_type="event",
            source_name=self.source_name,
            external_id=external_id,
            title=title,
            description=final_description,
            url=website or self.source_url,
            published_at=None,
            deadline_at=deadline_at,
            city="Красногорск",
            region="Московская область",
            customer_name=organizer,
            event_name=title,
            venue_name=venue or "Крокус Экспо",
            raw_payload={
                "date_range": block[0],
                "organizer": organizer,
                "website": website,
                "source_url": self.source_url,
            },
        )

    def _find_block_end(self, lines: list[str], start_index: int) -> int:
        for idx in range(start_index, len(lines)):
            if self._is_date_range(lines[idx]):
                return idx
        return len(lines)

    def _extract_title_and_description(self, lines: list[str], date_index: int) -> tuple[str | None, str | None]:
        title: str | None = None
        description: str | None = None

        if date_index >= 2 and not self._looks_like_noise(lines[date_index - 2]):
            title = lines[date_index - 2]
            if not self._looks_like_noise(lines[date_index - 1]):
                description = lines[date_index - 1]
        elif date_index >= 1 and not self._looks_like_noise(lines[date_index - 1]):
            title = lines[date_index - 1]

        return title, description

    def _extract_deadline(self, date_range: str) -> datetime | None:
        if "—" not in date_range and "-" not in date_range:
            return None
        parts = re.split(r"[—-]", date_range, maxsplit=1)
        if len(parts) != 2:
            return None
        return self._parse_russian_date(parts[1].strip())

    def _extract_field(self, block: list[str], label: str) -> str | None:
        for idx, line in enumerate(block):
            if line.startswith(label):
                value = line.replace(label, "", 1).strip()
                if value:
                    return value
                if idx + 1 < len(block):
                    return block[idx + 1]
        return None

    def _extract_website(self, block: list[str]) -> str | None:
        for line in block:
            if line.startswith("Сайт:"):
                candidate = line.replace("Сайт:", "", 1).strip()
                if not candidate:
                    continue
                candidate = candidate.split()[0]
                if candidate.startswith("www."):
                    return f"https://{candidate}"
                if candidate.startswith("http"):
                    return candidate
        return None

    def _extract_summary(self, block: list[str]) -> str | None:
        for idx, line in enumerate(block):
            if line == "Краткое описание:":
                summary_lines: list[str] = []
                for inner in block[idx + 1 :]:
                    if inner.startswith("Онлайн-регистрация") or inner.startswith("Контактная информация:"):
                        break
                    summary_lines.append(inner)
                return " ".join(summary_lines).strip() or None
        return None

    def _make_external_id(self, title: str, date_range: str) -> str:
        digest = hashlib.md5(f"{title}|{date_range}".encode("utf-8")).hexdigest()
        return digest

    def _is_date_range(self, value: str) -> bool:
        return bool(re.search(r"\d{1,2}\s+[А-Яа-яA-Za-z]+\s+\d{4}\s+[—-]\s+\d{1,2}\s+[А-Яа-яA-Za-z]+\s+\d{4}", value))

    def _looks_like_noise(self, value: str) -> bool:
        noise_prefixes = (
            "Image:",
            "###",
            "*",
            "О выставке:",
            "Краткое описание:",
            "Контактная информация:",
        )
        return not value or value.startswith(noise_prefixes)

    def _clean_line(self, value: str) -> str:
        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def _normalize_response_encoding(self, response: requests.Response) -> None:
        # Some sources omit or misreport charset; prefer the detected one when needed.
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding or "utf-8"

    def _parse_russian_date(self, value: str) -> datetime | None:
        match = re.search(r"(\d{1,2})\s+([А-Яа-яA-Za-z]+)\s+(\d{4})", value)
        if not match:
            return None

        day = int(match.group(1))
        month_name = match.group(2).lower()
        year = int(match.group(3))
        month = self.MONTHS.get(month_name)
        if month is None:
            return None

        try:
            return datetime(year, month, day)
        except ValueError:
            return None
