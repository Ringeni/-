"""bootstrap migration

Revision ID: 20260405_0001
Revises:
Create Date: 2026-04-05 20:00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260405_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Bootstrap revision to validate migration workflow in week 4.
    pass


def downgrade() -> None:
    pass
