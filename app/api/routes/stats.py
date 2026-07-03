from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.stats import DailyStats, HotLeadStats
from app.services.lead_service import LeadService

router = APIRouter()


@router.get("/daily", response_model=DailyStats)
def daily_stats(db: Session = Depends(get_db)) -> DailyStats:
    return LeadService(db).daily_stats()


@router.get("/hot", response_model=HotLeadStats)
def hot_stats(db: Session = Depends(get_db)) -> HotLeadStats:
    return LeadService(db).hot_stats()
