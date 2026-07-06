import logging

from app.collectors.events.crocus_expo import CrocusExpoCollector
from app.collectors.events.exponet_city import ExponetCityCollector
from app.collectors.events.expocentr import ExpocentrCollector
from app.collectors.tenders.b2b_center import B2BCenterCollector
from app.collectors.tenders.bidzaar import BidzaarCollector
from app.collectors.tenders.eis import EisCollector
from app.collectors.tenders.fabrikant import FabrikantCollector
from app.collectors.tenders.rostender import RostenderCollector
from app.collectors.tenders.synapse import SynapseCollector
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.collector_service import CollectorService

logger = logging.getLogger(__name__)

EXPONET_CITY_SOURCES = [
    ("moscow", "Москва", "Москва"),
    ("spb", "Санкт-Петербург", "Санкт-Петербург"),
    ("ekaterinburg", "Екатеринбург", "Свердловская область"),
    ("kazan", "Казань", "Республика Татарстан"),
    ("novosibirsk", "Новосибирск", "Новосибирская область"),
    ("krasnodar", "Краснодар", "Краснодарский край"),
    ("sochi", "Сочи", "Краснодарский край"),
    ("samara", "Самара", "Самарская область"),
    ("ufa", "Уфа", "Республика Башкортостан"),
    ("rostov", "Ростов-на-Дону", "Ростовская область"),
    ("perm", "Пермь", "Пермский край"),
    ("chelyabinsk", "Челябинск", "Челябинская область"),
    ("mineralwater", "Минеральные Воды", "Ставропольский край"),
    ("pyatigorsk", "Пятигорск", "Ставропольский край"),
    ("krasnoyarsk", "Красноярск", "Красноярский край"),
    ("irkutsk", "Иркутск", "Иркутская область"),
]


class RunnerService:
    def __init__(self) -> None:
        settings = get_settings()
        self.collectors = []
        if settings.enable_crocus_collector:
            self.collectors.append(CrocusExpoCollector())
        if settings.enable_exponet_city_collectors:
            self.collectors.extend(
                ExponetCityCollector(city_slug, city_name, region_name)
                for city_slug, city_name, region_name in EXPONET_CITY_SOURCES
            )
        if settings.enable_expocentr_collector:
            self.collectors.append(ExpocentrCollector())
        if settings.enable_b2b_center_collector:
            self.collectors.append(B2BCenterCollector())
        if settings.enable_bidzaar_collector:
            self.collectors.append(BidzaarCollector())
        if settings.enable_fabrikant_collector:
            self.collectors.append(FabrikantCollector())
        if settings.enable_rostender_collector:
            self.collectors.append(RostenderCollector())
        if settings.enable_synapse_collector:
            self.collectors.append(SynapseCollector())
        if settings.enable_eis_collector:
            self.collectors.append(EisCollector())

    def run_all(self) -> list[tuple[str, int, int, str]]:
        results: list[tuple[str, int, int, str]] = []
        with SessionLocal() as db:
            service = CollectorService(db)
            for collector in self.collectors:
                try:
                    items = collector.fetch()
                    found, saved = service.run_collector(collector.source_name, items)
                    results.append((collector.source_name, found, saved, "success"))
                except Exception as exc:
                    safe_error = self._safe_error(exc)
                    logger.warning(
                        "Collector failed",
                        extra={"collector": collector.source_name, "error": safe_error},
                    )
                    results.append((collector.source_name, 0, 0, f"failed: {safe_error}"))
        return results

    def _safe_error(self, exc: Exception) -> str:
        settings = get_settings()
        value = str(exc)
        if settings.telegram_bot_token:
            value = value.replace(settings.telegram_bot_token, "<telegram-token>")
        return value
