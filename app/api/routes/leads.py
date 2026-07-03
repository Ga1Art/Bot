from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.schemas.lead import LeadRead, LeadStatusUpdate
from app.services.export_service import ExportService
from app.services.google_sheets_service import GoogleSheetsService
from app.services.lead_service import LeadService

router = APIRouter()


@router.get("", response_model=list[LeadRead])
def list_leads(
    status: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    region: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
    hot_only: bool = Query(default=False),
    limit: int | None = Query(default=None, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[LeadRead]:
    return LeadService(db).list_leads(
        status=status,
        priority=priority,
        region=region,
        source_type=source_type,
        hot_only=hot_only,
        limit=limit,
    )


@router.get("/queue", response_model=list[LeadRead])
def review_queue(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[LeadRead]:
    return LeadService(db).review_queue(limit=limit)


@router.get("/hot", response_model=list[LeadRead])
def hot_prospects(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[LeadRead]:
    return LeadService(db).hot_prospects(limit=limit)


@router.get("/export")
def export_leads(
    status: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    region: str | None = Query(default=None),
    source_type: str | None = Query(default=None),
    hot_only: bool = Query(default=False),
    limit: int | None = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> Response:
    leads = LeadService(db).list_leads(
        status=status,
        priority=priority,
        region=region,
        source_type=source_type,
        hot_only=hot_only,
        limit=limit,
    )
    csv_content = ExportService().leads_to_csv(leads)
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="leads_export.csv"'},
    )


@router.get("/export/hot")
def export_hot_leads(
    limit: int | None = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> Response:
    leads = LeadService(db).hot_prospects(limit=limit)
    csv_content = ExportService().leads_to_csv(leads)
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="hot_leads_export.csv"'},
    )


@router.post("/sync-sheets")
def sync_sheets(
    limit: int | None = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> dict[str, int | str]:
    leads = LeadService(db).review_queue(limit=limit)
    settings = get_settings()
    synced = GoogleSheetsService().sync_leads(leads, range_name=settings.google_sheets_range)
    return {"status": "ok", "synced": synced}


@router.post("/sync-sheets/hot")
def sync_hot_sheets(
    limit: int | None = Query(default=500, ge=1, le=5000),
    db: Session = Depends(get_db),
) -> dict[str, int | str]:
    leads = LeadService(db).hot_prospects(limit=limit)
    settings = get_settings()
    synced = GoogleSheetsService().sync_leads(leads, range_name=settings.google_sheets_hot_range)
    return {"status": "ok", "synced": synced}


@router.patch("/{lead_id}/status", response_model=LeadRead)
def update_status(
    lead_id: str,
    payload: LeadStatusUpdate,
    db: Session = Depends(get_db),
) -> LeadRead:
    return LeadService(db).update_status(lead_id=lead_id, payload=payload)
