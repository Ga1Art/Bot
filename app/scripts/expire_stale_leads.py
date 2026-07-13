from app.db.session import SessionLocal
from app.repositories.lead_repo import LeadRepository


def main() -> None:
    with SessionLocal() as db:
        expired = LeadRepository(db).expire_stale_open_leads()
        db.commit()
    print(f"Expired stale queue leads: {expired}")


if __name__ == "__main__":
    main()
