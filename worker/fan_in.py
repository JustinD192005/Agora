"""Fan-in coordination: deciding when all researchers are done.

When a researcher finishes, it calls `try_enqueue_synthesizer(run_id, redis)`.
The function acquires a row-level lock on the parent run, counts completed
researcher tasks, and enqueues the synthesizer exactly once — even if
multiple researchers finish simultaneously.

The row lock is the key primitive. Without it, two researchers finishing at
the same instant could both conclude "I'm last" and fire the synthesizer
twice. With it, Postgres serializes their completion logic through the
FOR UPDATE lock, so they take turns checking-and-acting atomically.
"""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from sqlalchemy import Integer, select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: F401  (future use)
from sqlalchemy.ext.asyncio import AsyncSession

from api.db import Event, Run, SessionLocal, Task

log = structlog.get_logger()


# Researcher terminal states — any of these count as "this researcher is done"
TERMINAL_RESEARCHER_STATUSES = ("completed", "completed_partial", "failed")


async def try_enqueue_synthesizer(run_id: str, redis) -> bool:
    """Check if all researchers are done; if so, enqueue the synthesizer.

    This is the fan-in coordination point. Called by every researcher after
    its own task row has been committed. Uses a row-level lock on `runs` so
    concurrent callers serialize safely.

    Returns:
        True if THIS call enqueued the synthesizer.
        False if synthesizer was already enqueued, or not all researchers
        are done yet.
    """
    run_uuid = UUID(run_id)

    async with SessionLocal() as session:
        # --- Acquire exclusive row lock on the run ---
        # Anyone else trying to do the same for this run_id blocks here
        # until we commit or rollback. Postgres guarantees only one
        # transaction at a time holds this lock.
        lock_stmt = (
            select(Run)
            .where(Run.id == run_uuid)
            .with_for_update()  # SQL: SELECT ... FOR UPDATE
        )
        result = await session.execute(lock_stmt)
        run = result.scalar_one_or_none()

        if run is None:
            log.error("fan_in.run_not_found", run_id=run_id)
            return False

        # --- Guard: was the synthesizer already enqueued? ---
        if run.synthesizer_enqueued_at is not None:
            log.info(
                "fan_in.already_enqueued",
                run_id=run_id,
                enqueued_at=run.synthesizer_enqueued_at.isoformat(),
            )
            return False

        # --- Guard: do we even know how many researchers to expect? ---
        if run.expected_researchers is None:
            log.warning("fan_in.no_expected_count", run_id=run_id)
            return False

        # --- Count terminal researcher tasks (by distinct sub_question_index) ---
        # Using DISTINCT sub_question_index defends against the case where
        # multiple task rows exist for the same sub-question (e.g. if arq
        # retries somehow created duplicates). We care about "how many
        # distinct sub-questions have reached terminal status", not "how
        # many task rows exist". With max_tries=1 on WorkerSettings this
        # should be a no-op, but it's defense in depth.
        count_stmt = (
            select(func.count(func.distinct(Task.input["sub_question_index"].astext.cast(Integer))))
            .select_from(Task)
            .where(
                Task.run_id == run_uuid,
                Task.kind == "researcher",
                Task.status.in_(TERMINAL_RESEARCHER_STATUSES),
            )
        )
        result = await session.execute(count_stmt)
        completed = result.scalar_one()

        log.info(
            "fan_in.check",
            run_id=run_id,
            completed=completed,
            expected=run.expected_researchers,
        )

        if completed < run.expected_researchers:
            # Not everyone's in yet. Release the lock by committing
            # (nothing was changed). Other researchers finishing later
            # will run this same check and maybe be the one who triggers.
            return False

        # --- All researchers done. We're the trigger. ---
        # Mark the run so no one else tries (even after we release the lock).
        run.synthesizer_enqueued_at = datetime.now(timezone.utc)
        run.status = "synthesizing"

        session.add(Event(
            run_id=run_uuid,
            kind="synthesizer_enqueueing",
            payload={
                "completed_researchers": completed,
                "expected_researchers": run.expected_researchers,
            },
        ))

        await session.commit()
        # Lock released here via commit.

    # --- Enqueue the synthesizer OUTSIDE the DB transaction ---
    # Same reasoning as the planner: commit first, enqueue second.
    # If the enqueue fails, synthesizer_enqueued_at is still set and
    # the run is stuck in 'synthesizing' — we accept that failure mode
    # for today (Day 5 concern).
    job = await redis.enqueue_job("run_synthesizer", run_id)
    log.info("fan_in.synthesizer_enqueued", run_id=run_id, job_id=job.job_id)

    return True