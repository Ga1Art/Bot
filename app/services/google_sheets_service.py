from __future__ import annotations

import logging
from decimal import Decimal

from app.core.config import get_settings
from app.schemas.lead import LeadRead

logger = logging.getLogger(__name__)


class GoogleSheetsService:
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self) -> None:
        self.settings = get_settings()

    def is_configured(self) -> bool:
        return bool(
            self.settings.google_sheets_credentials_file
            and self.settings.google_sheets_spreadsheet_id
            and self.settings.google_sheets_range
        )

    def sync_leads(self, leads: list[LeadRead], range_name: str | None = None) -> int:
        if not self.is_configured():
            logger.info("Google Sheets is not configured; sync skipped")
            return 0

        service = self._build_service()
        target_range = range_name or self.settings.google_sheets_range
        values = self._build_values(leads)
        body = {"values": values}
        service.spreadsheets().values().clear(
            spreadsheetId=self.settings.google_sheets_spreadsheet_id,
            range=self._sheet_name_from_range(target_range),
            body={},
        ).execute()
        service.spreadsheets().values().update(
            spreadsheetId=self.settings.google_sheets_spreadsheet_id,
            range=target_range,
            valueInputOption="RAW",
            body=body,
        ).execute()
        return max(len(values) - 1, 0)

    def _build_service(self):
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_service_account_file(
            self.settings.google_sheets_credentials_file,
            scopes=self.SCOPES,
        )
        return build("sheets", "v4", credentials=credentials, cache_discovery=False)

    def _build_values(self, leads: list[LeadRead]) -> list[list[str]]:
        rows: list[list[str]] = [
            [
                "id",
                "priority",
                "status",
                "source_type",
                "source_name",
                "title",
                "customer_name",
                "region",
                "budget_max",
                "deadline_at",
                "relevance_score",
                "url",
            ]
        ]

        for lead in leads:
            rows.append(
                [
                    str(lead.id),
                    lead.priority,
                    lead.status,
                    lead.source_type,
                    lead.source_name,
                    lead.title,
                    lead.customer_name or "",
                    lead.region or "",
                    self._format_budget(lead.budget_max),
                    lead.deadline_at.strftime("%d.%m.%Y %H:%M") if lead.deadline_at else "",
                    str(lead.relevance_score),
                    lead.url,
                ]
            )

        return rows

    def _format_budget(self, value: Decimal | None) -> str:
        if value is None:
            return ""
        return str(value)

    def _sheet_name_from_range(self, range_name: str) -> str:
        if "!" not in range_name:
            return range_name
        return range_name.split("!", maxsplit=1)[0]
