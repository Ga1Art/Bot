import argparse

from app.db.session import SessionLocal
from app.core.config import get_settings
from app.services.google_sheets_service import GoogleSheetsService
from app.services.lead_service import LeadService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync leads to Google Sheets.")
    parser.add_argument("--hot", action="store_true", help="Sync only hot prospects.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of leads to sync.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    with SessionLocal() as db:
        service = LeadService(db)
        leads = service.hot_prospects(limit=args.limit) if args.hot else service.review_queue(limit=args.limit)

    range_name = settings.google_sheets_hot_range if args.hot else settings.google_sheets_range
    synced = GoogleSheetsService().sync_leads(leads, range_name=range_name)
    print(f"Synced rows={synced}")


if __name__ == "__main__":
    main()
