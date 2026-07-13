"""add quality components and duplicate reason

Revision ID: 0005_add_quality_components
Revises: 0004_add_ai_learning_fields
Create Date: 2026-07-13 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0005_add_quality_components"
down_revision: Union[str, None] = "0004_add_ai_learning_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("leads", sa.Column("fit_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("business_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("urgency_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("logistics_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("quality_reason", sa.Text(), nullable=True))
    op.add_column("leads", sa.Column("duplicate_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("leads", "duplicate_reason")
    op.drop_column("leads", "quality_reason")
    op.drop_column("leads", "logistics_score")
    op.drop_column("leads", "urgency_score")
    op.drop_column("leads", "business_score")
    op.drop_column("leads", "fit_score")
