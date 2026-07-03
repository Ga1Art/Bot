from app.db.session import SessionLocal
from app.services.google_sheets_service import GoogleSheetsService
from app.services.lead_service import LeadService


class SyncService:
    def sync_queue_to_google_sheets(self, limit: int = 500) -> int:
        with SessionLocal() as db:
            leads = LeadService(db).review_queue(limit=limit)
        return GoogleSheetsService().sync_leads(leads)
