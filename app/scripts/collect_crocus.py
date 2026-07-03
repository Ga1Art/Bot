from app.collectors.events.crocus_expo import CrocusExpoCollector
from app.db.session import SessionLocal
from app.services.collector_service import CollectorService


def main() -> None:
    collector = CrocusExpoCollector()
    items = collector.fetch()

    with SessionLocal() as db:
        found, saved = CollectorService(db).ingest(items)

    print(f"Collected={found} Saved={saved}")


if __name__ == "__main__":
    main()
