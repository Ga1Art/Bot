from __future__ import annotations

import csv
import io

from app.schemas.lead import LeadRead


class ExportService:
    def leads_to_csv(self, leads: list[LeadRead]) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=";")
        writer.writerow(
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
        )

        for lead in leads:
            writer.writerow(
                [
                    lead.id,
                    lead.priority,
                    lead.status,
                    lead.source_type,
                    lead.source_name,
                    lead.title,
                    lead.customer_name or "",
                    lead.region or "",
                    str(lead.budget_max) if lead.budget_max is not None else "",
                    lead.deadline_at.strftime("%d.%m.%Y %H:%M") if lead.deadline_at else "",
                    lead.relevance_score,
                    lead.url,
                ]
            )

        return buffer.getvalue()
