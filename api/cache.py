"""Deterministic replay cache.

Caches LLM responses and tool results keyed by the hash of their normalized
input. Hit means we've seen identical input before and can return the prior
output without hitting the network.

Design notes:
- Normalization: JSON-serialize with sort_keys=True and no whitespace, so
  logically-equivalent inputs (same dict, different key order) hash identically.
- Hashing: SHA-256. Overkill for collision avoidance but standard and fast.
- TTL: LLM calls cache indefinitely. Tool results (especially web_fetch) get
  a 24-hour TTL so stale web content eventually re-fetches.
- Scope: shared across all users/runs. If two people ask "compare X and Y",
  the second one gets the first one's researcher output for free.
- The API endpoint can pass bust_cache=True to disable lookups for a run,
  which forces fresh LLM calls (useful when testing prompt changes).
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.db import LLMCache, SessionLocal

log = structlog.get_logger()


# Default TTLs
LLM_TTL: timedelta | None = None  # LLM calls never expire
TOOL_TTL_WEB_FETCH = timedelta(hours=24)  # web content goes stale
TOOL_TTL_WEB_SEARCH = timedelta(hours=6)  # search rankings drift faster


def hash_input(payload: object) -> str:
    """Produce a deterministic SHA-256 hash of an arbitrary JSON-serializable payload.

    JSON is serialized with sort_keys=True and no whitespace, so logically
    equivalent inputs produce identical hashes regardless of key ordering or
    formatting of the original data.
    """
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def get_cached(input_hash: str, kind: str, model: str) -> dict | None:
    """Return the cached output for this (hash, kind, model) triple, or None on miss.

    Also returns None if the entry has expired (and the row is left alone —
    cleanup is a separate concern, not on the hot path).
    """
    async with SessionLocal() as session:
        stmt = select(LLMCache).where(
            LLMCache.input_hash == input_hash,
            LLMCache.kind == kind,
            LLMCache.model == model,
        )
        result = await session.execute(stmt)
        entry = result.scalar_one_or_none()
        if entry is None:
            log.debug("cache.miss", kind=kind, model=model, hash=input_hash[:12])
            return None

        # Check TTL
        if entry.expires_at is not None and entry.expires_at < datetime.now(timezone.utc):
            log.debug("cache.expired", kind=kind, model=model, hash=input_hash[:12])
            return None

        log.info("cache.hit", kind=kind, model=model, hash=input_hash[:12])
        return entry.output


async def set_cached(
    input_hash: str,
    kind: str,
    model: str,
    output: dict,
    ttl: timedelta | None = None,
) -> None:
    """Store output in the cache. Overwrites any existing entry for the same (hash, kind, model).

    Uses Postgres INSERT ... ON CONFLICT DO UPDATE for atomic upsert, so concurrent
    cache writes for the same key don't crash.
    """
    expires_at = (
        datetime.now(timezone.utc) + ttl if ttl is not None else None
    )

    async with SessionLocal() as session:
        stmt = pg_insert(LLMCache).values(
            input_hash=input_hash,
            kind=kind,
            model=model,
            output=output,
            expires_at=expires_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["input_hash", "kind", "model"],
            set_={
                "output": stmt.excluded.output,
                "expires_at": stmt.excluded.expires_at,
                "created_at": datetime.now(timezone.utc),
            },
        )
        await session.execute(stmt)
        await session.commit()
        log.debug("cache.set", kind=kind, model=model, hash=input_hash[:12])


async def clear_cache() -> int:
    """Delete ALL cache entries. Returns the number of rows deleted.

    Useful for testing and for manual cache invalidation during development.
    Not exposed via HTTP — direct DB call only.
    """
    from sqlalchemy import delete

    async with SessionLocal() as session:
        result = await session.execute(delete(LLMCache))
        await session.commit()
        return result.rowcount or 0