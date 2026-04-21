"""Arq worker — consumes jobs from Redis and processes them."""
from datetime import datetime, timezone
from uuid import UUID

import structlog
from arq.connections import RedisSettings
from sqlalchemy import select

from api.config import get_settings
from api.db import Event, Run, SessionLocal

log = structlog.get_logger()
settings = get_settings()


async def run_planner(ctx: dict, run_id: str) -> None:
    """Day 1 stub — just marks the run as 'planning' and logs an event.

    In Day 2 we'll actually call Gemini here and produce a real plan.
    """
    run_uuid = UUID(run_id)
    log.info("planner.start", run_id=run_id)

    async with SessionLocal() as session:
        result = await session.execute(select(Run).where(Run.id == run_uuid))
        run = result.scalar_one_or_none()
        if run is None:
            log.error("planner.run_not_found", run_id=run_id)
            return

        run.status = "planning"
        session.add(Event(
            run_id=run_uuid,
            kind="planner_started",
            payload={"message": "Day 1 stub — real planner lands Day 2"},
        ))
        await session.commit()

    log.info("planner.done", run_id=run_id)


class WorkerSettings:
    """Arq worker configuration — arq imports this class by path."""
    functions = [run_planner]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10
    job_timeout = 60