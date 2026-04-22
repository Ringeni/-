"""week7 warning fields + event table

Revision ID: 20260422_0004
Revises: 20260406_0003
Create Date: 2026-04-22 10:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260422_0004"
down_revision: Union[str, None] = "20260406_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to warning table
    op.add_column("warning", sa.Column("handler_name", sa.Text(), nullable=True))
    op.add_column("warning", sa.Column("handled_at", sa.DateTime(), nullable=True))
    op.add_column("warning", sa.Column("closed_at", sa.DateTime(), nullable=True))
    op.add_column("warning", sa.Column("closed_by", sa.Integer(), nullable=True))
    op.add_column("warning", sa.Column("closed_by_name", sa.Text(), nullable=True))

    # Create event table
    # Note: stream_id uses VARCHAR(64) instead of TEXT for index compatibility
    op.create_table(
        "event",
        sa.Column("event_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("stream_id", sa.String(64), nullable=False),
        sa.Column("event_type", sa.Integer(), nullable=False),
        sa.Column("event_level", sa.Integer(), server_default="1", nullable=False),
        sa.Column("event_time", sa.DateTime(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("related_warning_id", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_event_target_id", "event", ["target_id"], unique=False)
    op.create_index("ix_event_target_time", "event", ["target_id", "event_time"], unique=False)
    op.create_index("ix_event_stream_time_type", "event", ["stream_id", "event_time", "event_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_event_stream_time_type", table_name="event")
    op.drop_index("ix_event_target_time", table_name="event")
    op.drop_index("ix_event_target_id", table_name="event")
    op.drop_table("event")

    op.drop_column("warning", "closed_by_name")
    op.drop_column("warning", "closed_by")
    op.drop_column("warning", "closed_at")
    op.drop_column("warning", "handled_at")
    op.drop_column("warning", "handler_name")