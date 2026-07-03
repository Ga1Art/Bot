import argparse

from app.db.models import Lead
from app.db.session import SessionLocal
from app.scoring.engine import explain_score_lead


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview lead scoring with factor breakdown.")
    parser.add_argument("--limit", type=int, default=10, help="How many leads to show.")
    parser.add_argument(
        "--sort",
        choices=["recent", "score"],
        default="recent",
        help="Sort by recent leads or by highest score.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with SessionLocal() as db:
        query = db.query(Lead)
        if args.sort == "score":
            query = query.order_by(Lead.relevance_score.desc(), Lead.created_at.desc())
        else:
            query = query.order_by(Lead.created_at.desc())

        leads = query.limit(args.limit).all()

    for index, lead in enumerate(leads, start=1):
        details = explain_score_lead(
            title=lead.title,
            description=lead.description,
            region=lead.region,
            budget_max=float(lead.budget_max) if lead.budget_max is not None else None,
        )
        print(f"[{index}] {lead.title}")
        print(f"    Source: {lead.source_name}")
        print(f"    Region: {lead.region or '-'}")
        print(f"    Score: {details.score} ({details.priority})")
        print(f"    Budget: {lead.budget_max if lead.budget_max is not None else '-'}")
        if details.factors:
            print("    Factors:")
            for factor in details.factors:
                print(f"      - {factor}")
        else:
            print("    Factors: none")
        print()


if __name__ == "__main__":
    main()
