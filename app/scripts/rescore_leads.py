from app.db.models import Lead
from app.db.session import SessionLocal
from app.core.config import get_settings
from app.schemas.collector import LeadCreate
from app.scoring.components import score_components
from app.services.ai_scoring_service import AiScoringService
from app.services.feedback_learning_service import FeedbackLearningService


def main() -> None:
    with SessionLocal() as db:
        settings = get_settings()
        leads = db.query(Lead).all()
        learning_service = FeedbackLearningService(db)
        ai_service = AiScoringService()
        updated = 0

        for lead in leads:
            components = score_components(
                title=lead.title,
                description=lead.description,
                region=lead.region,
                budget_max=float(lead.budget_max) if lead.budget_max is not None else None,
                deadline_at=lead.deadline_at,
            )
            payload = LeadCreate(
                source_type=lead.source_type,
                source_name=lead.source_name,
                external_id=lead.external_id,
                title=lead.title,
                description=lead.description,
                url=lead.url,
                published_at=lead.published_at,
                deadline_at=lead.deadline_at,
                city=lead.city,
                region=lead.region,
                budget_min=lead.budget_min,
                budget_max=lead.budget_max,
                currency=lead.currency,
                customer_name=lead.customer_name,
                event_name=lead.event_name,
                venue_name=lead.venue_name,
                keywords_matched=lead.keywords_matched,
                raw_payload=lead.raw_payload,
            )
            learning = learning_service.score_adjustment_for(payload)
            ai_adjustment = ai_service.score_adjustment_from_score(lead.ai_score)
            score = components.base_score + learning.value + ai_adjustment
            priority = (
                "A"
                if score >= settings.scoring_priority_a_threshold
                else "B"
                if score >= settings.scoring_priority_b_threshold
                else "C"
            )
            if (
                lead.relevance_score != score
                or lead.priority != priority
                or lead.base_relevance_score != components.base_score
                or lead.fit_score != components.fit_score
                or lead.business_score != components.business_score
                or lead.urgency_score != components.urgency_score
                or lead.logistics_score != components.logistics_score
                or lead.quality_reason != components.quality_reason
                or lead.learned_score_adjustment != learning.value
            ):
                lead.base_relevance_score = components.base_score
                lead.fit_score = components.fit_score
                lead.business_score = components.business_score
                lead.urgency_score = components.urgency_score
                lead.logistics_score = components.logistics_score
                lead.quality_reason = components.quality_reason
                lead.learned_score_adjustment = learning.value
                lead.learned_reason = learning.reason
                lead.relevance_score = score
                lead.priority = priority
                updated += 1

        db.commit()

    print(f"Rescored={updated} Total={len(leads)}")


if __name__ == "__main__":
    main()
