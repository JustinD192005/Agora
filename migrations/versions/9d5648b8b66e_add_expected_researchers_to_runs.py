"""add expected_researchers to runs

Revision ID: 9d5648b8b66e
Revises: 49e5177da2d6
Create Date: 2026-04-22

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "9d5648b8b66e"
down_revision: Union[str, None] = "49e5177da2d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add a column to track how many researchers a run expects.

    This is set by the planner once the plan is generated. The fan-in logic
    compares completed researcher count against this target to decide when
    to trigger the synthesizer.
    """
    op.add_column(
        "runs",
        sa.Column("expected_researchers", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "expected_researchers")