from app.collectors.tenders.eis import EisCollector
from app.db.session import SessionLocal
from app.services.collector_service import CollectorService


def main() -> None:
    collector = EisCollector()
    items = collector.fetch()

    with SessionLocal() as db:
        found, saved = CollectorService(db).run_collector(collector.source_name, items)

    print(f"Collected={found} Saved={saved}")


if __name__ == "__main__":
    main()
