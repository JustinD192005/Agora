"""add llm_cache table

Revision ID: b88e96bace69
Revises: 04a574e6eebd
Create Date: 2026-04-25

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "b88e96bace69"
down_revision: Union[str, None] = "04a574e6eebd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the llm_cache table for deterministic replay.

    Stores (input_hash, kind, model) → cached output JSON. Any LLM call or
    tool invocation can check this table before hitting the network; a hit
    means we've seen identical input before and can return the prior output.

    Columns:
    - input_hash: SHA-256 of the normalized input (prompt / messages / tool input)
    - kind: "llm" | "tool" — distinguishes LLM responses from tool results
    - model: the model name or tool name (e.g. "gemini-2.5-flash", "web_search")
    - output: the cached output as JSONB
    - created_at: when this entry was cached
    - expires_at: optional TTL; NULL means "never expires"

    Primary key is (input_hash, kind, model) — the same hash with a different
    model/tool is a different cache entry.
    """
    op.create_table(
        "llm_cache",
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("output", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("input_hash", "kind", "model"),
    )

    # Index on expires_at for cleanup queries
    op.create_index(
        "ix_llm_cache_expires_at",
        "llm_cache",
        ["expires_at"],
        postgresql_where=sa.text("expires_at IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_llm_cache_expires_at", table_name="llm_cache")
    op.drop_table("llm_cache")