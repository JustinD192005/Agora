"""Mark stuck runs as failed.

A run is "stuck" when it has been in a non-terminal status for longer than
--threshold-minutes. This typically means a worker crashed between marking
the run (e.g. as `synthesizing`) and actually finishing the work — exactly
the failure mode acknowledged in worker/fan_in.py.

Run periodically (cron, k8s CronJob, manual, etc.) to keep the runs table honest.

Usage:
    python -m scripts.reap_stuck_runs                          # threshold 15 min, real run
    python -m scripts.reap_stuck_runs --threshold-minutes 30
    python -m scripts.reap_stuck_runs --dry-run
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select

from api.db import Event, Run, SessionLocal

log = structlog.get_logger()


# Anything not in (completed, failed) is "still in flight" from the run's POV.
NON_TERMINAL_STATUSES = ("pending", "planning", "researching", "synthesizing")


async def reap(threshold: timedelta, dry_run: bool) -> int:
    """Find non-terminal runs older than the threshold and mark them failed.

    Returns the count of runs reaped (or that would have been, in dry-run mode).
    """
    now = datetime.now(timezone.utc)
    cutoff = now - threshold

    async with SessionLocal() as session:
        result = await session.execute(
            select(Run)
            .where(Run.status.in_(NON_TERMINAL_STATUSES))
            .where(Run.created_at < cutoff)
            .order_by(Run.created_at.asc())
        )
        stuck = list(result.scalars().all())

        if not stuck:
            log.info("reaper.none_stuck", cutoff=cutoff.isoformat())
            return 0

        log.info(
            "reaper.found",
            n=len(stuck),
            cutoff=cutoff.isoformat(),
            dry_run=dry_run,
        )
        for r in stuck:
            log.info(
                "reaper.stuck_run",
                id=str(r.id),
                status=r.status,
                age_minutes=round((now - r.created_at).total_seconds() / 60, 1),
                question=r.user_question[:80],
            )

        if dry_run:
            return len(stuck)

        threshold_minutes = int(threshold.total_seconds() / 60)
        for r in stuck:
            previous = r.status
            r.status = "failed"
            r.completed_at = now
            session.add(Event(
                run_id=r.id,
                kind="run_reaped",
                payload={
                    "previous_status": previous,
                    "reason": "stuck_past_threshold",
                    "threshold_minutes": threshold_minutes,
                },
            ))
        await session.commit()

    return len(stuck)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mark stuck Agora runs as failed.")
    p.add_argument(
        "--threshold-minutes",
        type=int,
        default=15,
        help="Runs older than this in non-terminal status are reaped (default: 15).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be reaped without modifying the DB.",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    n = await reap(timedelta(minutes=args.threshold_minutes), args.dry_run)
    verb = "Would reap" if args.dry_run else "Reaped"
    print(f"{verb} {n} stuck run(s) (threshold={args.threshold_minutes}m).")


if __name__ == "__main__":
    asyncio.run(main())
