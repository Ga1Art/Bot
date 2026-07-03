"""create core tables"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_create_core_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "leads",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_name", sa.String(length=128), nullable=False),
        sa.Column("external_id", sa.String(length=256), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("deadline_at", sa.DateTime(), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("region", sa.String(length=128), nullable=True),
        sa.Column("budget_min", sa.Numeric(14, 2), nullable=True),
        sa.Column("budget_max", sa.Numeric(14, 2), nullable=True),
        sa.Column("currency", sa.String(length=16), nullable=True),
        sa.Column("customer_name", sa.String(length=256), nullable=True),
        sa.Column("event_name", sa.String(length=256), nullable=True),
        sa.Column("venue_name", sa.String(length=256), nullable=True),
        sa.Column("keywords_matched", sa.Text(), nullable=True),
        sa.Column("relevance_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("priority", sa.String(length=8), nullable=False, server_default="C"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="new"),
        sa.Column("is_duplicate", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("duplicate_of_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_name", "external_id", name="uq_leads_source_external_id"),
    )
    op.create_index("ix_leads_status", "leads", ["status"])
    op.create_index("ix_leads_priority", "leads", ["priority"])
    op.create_index("ix_leads_region", "leads", ["region"])
    op.create_index("ix_leads_published_at", "leads", ["published_at"])

    op.create_table(
        "lead_activities",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lead_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_type", sa.String(length=32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["lead_id"], ["leads.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "companies",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("website", sa.Text(), nullable=True),
        sa.Column("phone", sa.String(length=64), nullable=True),
        sa.Column("email", sa.String(length=128), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("segment", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "collector_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collector_name", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("items_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_saved", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("collector_runs")
    op.drop_table("companies")
    op.drop_table("lead_activities")
    op.drop_index("ix_leads_published_at", table_name="leads")
    op.drop_index("ix_leads_region", table_name="leads")
    op.drop_index("ix_leads_priority", table_name="leads")
    op.drop_index("ix_leads_status", table_name="leads")
    op.drop_table("leads")
