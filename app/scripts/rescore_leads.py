from app.db.models import Lead
from app.db.session import SessionLocal
from app.scoring.engine import score_lead


def main() -> None:
    with SessionLocal() as db:
        leads = db.query(Lead).all()
        updated = 0

        for lead in leads:
            score, priority = score_lead(
                title=lead.title,
                description=lead.description,
                region=lead.region,
                budget_max=float(lead.budget_max) if lead.budget_max is not None else None,
            )
            if lead.relevance_score != score or lead.priority != priority:
                lead.relevance_score = score
                lead.priority = priority
                updated += 1

        db.commit()

    print(f"Rescored={updated} Total={len(leads)}")


if __name__ == "__main__":
    main()
