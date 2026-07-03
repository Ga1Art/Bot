from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.leads import router as leads_router
from app.api.routes.stats import router as stats_router
from app.core.logging import configure_logging
from app.core.scheduler import scheduler


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Lead Radar", lifespan=lifespan)
app.include_router(health_router)
app.include_router(leads_router, prefix="/leads", tags=["leads"])
app.include_router(stats_router, prefix="/stats", tags=["stats"])
