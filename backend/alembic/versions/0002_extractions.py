"""extraction outcomes: one row per report and registry entry

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extractions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "report_id",
            sa.String(36),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("key", sa.String(60), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("message", sa.Text),
        sa.Column("model", sa.String(100)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("report_id", "key"),
    )
    # Reports ingested before outcomes were recorded: an entry with datapoints was extracted.
    # An entry without any is left without a row (the UI shows it as "not recorded" and offers
    # a retry): nothing in the database says whether it was absent or failed.
    op.execute(
        """
        INSERT INTO extractions (report_id, key, status, model)
        SELECT report_id, key, 'extracted', MIN(model)
        FROM datapoints
        GROUP BY report_id, key
        """
    )


def downgrade() -> None:
    op.drop_table("extractions")
