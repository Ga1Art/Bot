from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build

from app.core.config import get_settings


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or prepare Google Sheets for Lead Radar.")
    parser.add_argument("--credentials", help="Path to Google service account JSON file.")
    parser.add_argument("--spreadsheet-id", help="Existing spreadsheet id. If omitted, a new spreadsheet is created.")
    parser.add_argument("--title", default="Lead Radar", help="New spreadsheet title.")
    parser.add_argument("--queue-sheet", default="Queue", help="Sheet tab for the main queue.")
    parser.add_argument("--hot-sheet", default="Hot", help="Sheet tab for hot prospects.")
    parser.add_argument("--share-with", help="Google account email to share a newly created spreadsheet with.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    credentials_raw = args.credentials or settings.google_sheets_credentials_file
    if not credentials_raw:
        raise SystemExit("Pass --credentials or fill GOOGLE_SHEETS_CREDENTIALS_FILE in .env")
    credentials_path = Path(credentials_raw)
    if not credentials_path.exists():
        raise SystemExit(f"Credentials file not found: {credentials_path}")

    info = json.loads(credentials_path.read_text(encoding="utf-8"))
    client_email = info.get("client_email", "")
    credentials = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    sheets = build("sheets", "v4", credentials=credentials, cache_discovery=False)

    spreadsheet_id = args.spreadsheet_id or settings.google_sheets_spreadsheet_id
    created = False
    try:
        if spreadsheet_id:
            spreadsheet = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        else:
            spreadsheet = sheets.spreadsheets().create(
                body={
                    "properties": {"title": args.title},
                    "sheets": [
                        {"properties": {"title": args.queue_sheet}},
                        {"properties": {"title": args.hot_sheet}},
                    ],
                },
                fields="spreadsheetId,spreadsheetUrl,sheets.properties.title",
            ).execute()
            spreadsheet_id = spreadsheet["spreadsheetId"]
            created = True
    except HttpError as exc:
        _raise_setup_error(exc, client_email, credentials_path)

    existing_titles = {
        sheet["properties"]["title"]
        for sheet in spreadsheet.get("sheets", [])
    }
    missing_titles = [
        title
        for title in (args.queue_sheet, args.hot_sheet)
        if title not in existing_titles
    ]
    if missing_titles:
        try:
            sheets.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [
                        {"addSheet": {"properties": {"title": title}}}
                        for title in missing_titles
                    ]
                },
            ).execute()
        except HttpError as exc:
            _raise_setup_error(exc, client_email, credentials_path, spreadsheet_id)

    spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
    if args.share_with:
        drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
        try:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={
                    "type": "user",
                    "role": "writer",
                    "emailAddress": args.share_with,
                },
                fields="id",
                sendNotificationEmail=False,
            ).execute()
        except HttpError as exc:
            _raise_setup_error(exc, client_email, credentials_path, spreadsheet_id)

    print("Google Sheets is ready.")
    print(f"Created: {'yes' if created else 'no'}")
    print(f"Service account: {client_email or '-'}")
    print(f"Spreadsheet URL: {spreadsheet_url}")
    print()
    print("Put these values into .env:")
    print(f"GOOGLE_SHEETS_CREDENTIALS_FILE={credentials_path}")
    print(f"GOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet_id}")
    print(f"GOOGLE_SHEETS_RANGE={args.queue_sheet}!A1")
    print(f"GOOGLE_SHEETS_HOT_RANGE={args.hot_sheet}!A1")
    if not args.share_with:
        print()
        print("If you created a new spreadsheet and want to see it in your Drive, rerun with:")
        print("  --share-with your.email@gmail.com")


def _raise_setup_error(
    exc: HttpError,
    client_email: str,
    credentials_path: Path,
    spreadsheet_id: str | None = None,
) -> None:
    if exc.resp.status != 403:
        raise exc

    lines = [
        "Google returned HTTP 403: the service account has no permission.",
        "",
        f"Service account email: {client_email or '-'}",
        f"Credentials file: {credentials_path}",
        "",
        "Fastest fix:",
        "1. Create a Google Sheet manually in your browser.",
        f"2. Share it with this service account as Editor: {client_email or '<client_email from JSON>'}",
        "3. Copy the spreadsheet id from the URL.",
        "4. Put it into .env as GOOGLE_SHEETS_SPREADSHEET_ID.",
        "5. Run: .\\run.ps1 -Action setupsheets",
        "",
        "Also verify in Google Cloud Console:",
        "- Google Sheets API is enabled.",
        "- Google Drive API is enabled.",
        "- The JSON file belongs to the same project where those APIs are enabled.",
    ]
    if spreadsheet_id:
        lines.insert(4, f"Spreadsheet id: {spreadsheet_id}")
    raise SystemExit("\n".join(lines)) from exc


if __name__ == "__main__":
    main()
