"""add ai learning fields to leads

Revision ID: 0004_add_ai_learning_fields
Revises: 0003_add_lead_queue_indexes
Create Date: 2026-07-06 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0004_add_ai_learning_fields"
down_revision: Union[str, None] = "0003_add_lead_queue_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("leads", sa.Column("base_relevance_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("learned_score_adjustment", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("leads", sa.Column("learned_reason", sa.Text(), nullable=True))
    op.add_column("leads", sa.Column("ai_score", sa.Integer(), nullable=True))
    op.add_column("leads", sa.Column("ai_reason", sa.Text(), nullable=True))
    op.add_column("leads", sa.Column("ai_recommended_action", sa.String(length=64), nullable=True))
    op.add_column("leads", sa.Column("ai_tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("leads", sa.Column("ai_risk_tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("leads", sa.Column("ai_model", sa.String(length=128), nullable=True))
    op.add_column("leads", sa.Column("ai_analyzed_at", sa.DateTime(), nullable=True))
    op.execute("UPDATE leads SET base_relevance_score = relevance_score WHERE base_relevance_score IS NULL")
    op.alter_column("leads", "learned_score_adjustment", server_default=None)


def downgrade() -> None:
    op.drop_column("leads", "ai_analyzed_at")
    op.drop_column("leads", "ai_model")
    op.drop_column("leads", "ai_risk_tags")
    op.drop_column("leads", "ai_tags")
    op.drop_column("leads", "ai_recommended_action")
    op.drop_column("leads", "ai_reason")
    op.drop_column("leads", "ai_score")
    op.drop_column("leads", "learned_reason")
    op.drop_column("leads", "learned_score_adjustment")
    op.drop_column("leads", "base_relevance_score")
