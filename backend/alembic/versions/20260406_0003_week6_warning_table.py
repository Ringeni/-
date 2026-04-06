"""week6 warning table

Revision ID: 20260406_0003
Revises: 20260405_0002
Create Date: 2026-04-06 09:30:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260406_0003"
down_revision: Union[str, None] = "20260405_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "warning",
        sa.Column("warning_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.BigInteger(), nullable=True),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("warning_level", sa.Integer(), server_default="1", nullable=False),
        sa.Column("trigger_time", sa.DateTime(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.Integer(), server_default="0", nullable=False),
        sa.Column("handler_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("warning_id"),
    )
    op.create_index("ix_warning_target_id", "warning", ["target_id"], unique=False)
    op.create_index("ix_warning_status_trigger_time", "warning", ["status", "trigger_time"], unique=False)
    op.create_index("ix_warning_target_handler", "warning", ["target_id", "handler_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_warning_target_handler", table_name="warning")
    op.drop_index("ix_warning_status_trigger_time", table_name="warning")
    op.drop_index("ix_warning_target_id", table_name="warning")
    op.drop_table("warning")
