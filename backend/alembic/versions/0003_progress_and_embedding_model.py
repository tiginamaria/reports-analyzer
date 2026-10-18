"""report progress detail and the embedding model of its vectors

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

# What every report was embedded with before the hosted option existed.
PREVIOUS_EMBEDDING = "local/BAAI/bge-m3"


def upgrade() -> None:
    op.add_column("reports", sa.Column("done", sa.Integer, nullable=False, server_default="0"))
    op.add_column("reports", sa.Column("total", sa.Integer, nullable=False, server_default="0"))
    op.add_column("reports", sa.Column("activity", sa.String(300)))
    op.add_column("reports", sa.Column("started_at", sa.DateTime(timezone=True)))
    op.add_column("reports", sa.Column("embedding_model", sa.String(120)))
    # Reports indexed so far hold local bge-m3 vectors. If the configured embedding model
    # differs they are not searched until `python -m app.reembed` rebuilds them.
    op.execute(
        f"UPDATE reports SET embedding_model = '{PREVIOUS_EMBEDDING}' "
        "WHERE status IN ('ready', 'extracting')"
    )


def downgrade() -> None:
    for column in ("embedding_model", "started_at", "activity", "total", "done"):
        op.drop_column("reports", column)
