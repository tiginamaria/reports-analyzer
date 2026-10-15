"""initial schema

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("company", sa.String(200), nullable=False, index=True),
        sa.Column("fiscal_year", sa.Integer, nullable=False, index=True),
        sa.Column("title", sa.String(300)),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("page_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "pages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "report_id",
            sa.String(36),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("page_no", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.UniqueConstraint("report_id", "page_no"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(80), primary_key=True),
        sa.Column(
            "report_id",
            sa.String(36),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(10), nullable=False, server_default="text"),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("word_count", sa.Integer, nullable=False, server_default="0"),
    )
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING GIN (tsv)")

    op.create_table(
        "datapoints",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "report_id",
            sa.String(36),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("key", sa.String(60), nullable=False, index=True),
        sa.Column("label", sa.String(300), nullable=False),
        sa.Column("value", sa.String(500)),
        sa.Column("numeric_value", sa.Float),
        sa.Column("unit", sa.String(100)),
        sa.Column("period", sa.String(100)),
        sa.Column("qualifiers", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("quote", sa.Text),
        sa.Column("chunk_id", sa.String(80)),
        sa.Column("page", sa.Integer),
        sa.Column("verification", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False, server_default="New conversation"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(36),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("status", sa.String(20)),
        sa.Column("citations", sa.JSON, nullable=False, server_default="[]"),
        sa.Column("meta", sa.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    for t in ("messages", "conversations", "datapoints", "chunks", "pages", "reports"):
        op.drop_table(t)
