"""add synthesizer_enqueued_at to runs

Revision ID: 04a574e6eebd
Revises: 9d5648b8b66e
Create Date: 2026-04-22

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "04a574e6eebd"
down_revision: Union[str, None] = "9d5648b8b66e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Record when the synthesizer was enqueued for a run.

    The fan-in logic (in each researcher's completion handler) checks this
    column under a row lock to decide whether to enqueue the synthesizer.
    If NULL, no synthesizer has been enqueued yet. If set, someone already
    fired it and this researcher should do nothing.
    """
    op.add_column(
        "runs",
        sa.Column("synthesizer_enqueued_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "synthesizer_enqueued_at")