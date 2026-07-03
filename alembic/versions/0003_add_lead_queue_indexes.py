"""add lead queue indexes"""

from alembic import op

revision = "0003_add_lead_queue_indexes"
down_revision = "0002_add_notified_at_to_leads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_leads_queue_lookup",
        "leads",
        ["status", "priority", "created_at"],
    )
    op.create_index(
        "ix_leads_hot_lookup",
        "leads",
        ["status", "priority", "region", "relevance_score", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_leads_hot_lookup", table_name="leads")
    op.drop_index("ix_leads_queue_lookup", table_name="leads")
