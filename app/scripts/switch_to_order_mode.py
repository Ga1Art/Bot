from app.db.models import Lead
from app.db.session import SessionLocal


def main() -> None:
    with SessionLocal() as db:
        updated = (
            db.query(Lead)
            .filter(Lead.source_type == "event")
            .filter(Lead.status.in_(("new", "in_work")))
            .update({Lead.status: "context"}, synchronize_session=False)
        )
        db.commit()

    print(f"Event context leads hidden from queue={updated}")


if __name__ == "__main__":
    main()
