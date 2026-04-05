"""week5 core tables

Revision ID: 20260405_0002
Revises: 20260405_0001
Create Date: 2026-04-05 22:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260405_0002"
down_revision: Union[str, None] = "20260405_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "camera",
        sa.Column("camera_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("location", sa.String(length=256), nullable=True),
        sa.Column("stream_url", sa.String(length=512), nullable=False),
        sa.Column("status", sa.Integer(), server_default="1", nullable=False),
        sa.Column("create_time", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column(
            "update_time",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("camera_id"),
    )

    op.create_table(
        "target",
        sa.Column("target_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("appearance_features", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("target_id"),
    )

    op.create_table(
        "user",
        sa.Column("user_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), server_default="viewer", nullable=False),
        sa.Column("email", sa.String(length=128), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("last_login_time", sa.DateTime(), nullable=True),
        sa.Column("create_time", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_user_username", "user", ["username"], unique=True)

    op.create_table(
        "trajectory",
        sa.Column("trajectory_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("stream_id", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("trajectory_id"),
    )
    op.create_index("ix_trajectory_target_id", "trajectory", ["target_id"], unique=False)
    op.create_index("ix_trajectory_stream_id", "trajectory", ["stream_id"], unique=False)
    op.create_index("ix_trajectory_target_start_time", "trajectory", ["target_id", "start_time"], unique=False)
    op.create_index("ix_trajectory_stream_start_time", "trajectory", ["stream_id", "start_time"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_trajectory_stream_start_time", table_name="trajectory")
    op.drop_index("ix_trajectory_target_start_time", table_name="trajectory")
    op.drop_index("ix_trajectory_stream_id", table_name="trajectory")
    op.drop_index("ix_trajectory_target_id", table_name="trajectory")
    op.drop_table("trajectory")

    op.drop_index("ix_user_username", table_name="user")
    op.drop_table("user")

    op.drop_table("target")
    op.drop_table("camera")
