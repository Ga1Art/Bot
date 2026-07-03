"""add notified_at to leads"""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_notified_at_to_leads"
down_revision = "0001_create_core_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("leads", sa.Column("notified_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("leads", "notified_at")
