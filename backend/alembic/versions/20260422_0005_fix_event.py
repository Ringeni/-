"""week8 fix event table indexes

Revision ID: 20260422_0005
Revises: 20260422_0004
Create Date: 2026-04-22 14:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260422_0005"
down_revision: Union[str, None] = "20260422_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Change stream_id from TEXT to VARCHAR(64) for index compatibility
    op.alter_column("event", "stream_id", existing_type=sa.Text(), type_=sa.String(64), existing_nullable=False)

    # Recreate indexes that may have failed
    op.create_index("ix_event_target_id", "event", ["target_id"], unique=False, if_not_exists=True)
    op.create_index("ix_event_target_time", "event", ["target_id", "event_time"], unique=False, if_not_exists=True)
    op.create_index("ix_event_stream_time_type", "event", ["stream_id", "event_time", "event_type"], unique=False, if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_event_stream_time_type", table_name="event")
    op.drop_index("ix_event_target_time", table_name="event")
    op.drop_index("ix_event_target_id", table_name="event")
    op.alter_column("event", "stream_id", existing_type=sa.String(64), type_=sa.Text(), existing_nullable=False)